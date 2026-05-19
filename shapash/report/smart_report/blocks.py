"""Block implementations and report data helpers for smart reports."""

from __future__ import annotations

import importlib
import importlib.metadata
from pathlib import Path
from uuid import uuid4

import pandas as pd
import plotly.express as px
import yaml

from shapash.plots.plot_evaluation_metrics import plot_confusion_matrix
from shapash.plots.plot_univariate import plot_distribution
from shapash.report.data_analysis import perform_global_dataframe_analysis
from shapash.report.smart_report.panel_support import render_plotly_pane_html
from shapash.report.smart_report.validation import stats_to_table
from shapash.utils.transform import apply_postprocessing, handle_categorical_missing, inverse_transform
from shapash.utils.utils import compute_sorted_variables_interactions_list_indices

PALETTE = {
    "gold": {"bg": "#ffffff", "border": "#f4c000", "title": "#f4c000", "text": "#343736"},
    "blue": {"bg": "#ffffff", "border": "#2255aa", "title": "#2255aa", "text": "#343736"},
    "gray": {"bg": "#ffffff", "border": "#eeeeee", "title": "#666666", "text": "#666666"},
    "orange": {"bg": "#fff9e6", "border": "#f4c000", "title": "#cc8833", "text": "#444444"},
}


class ReportBlockMixin:
    """Base mixin providing built-in and user-extensible smart report blocks.

    This mixin is consumed by the smart-report runtime to resolve YAML section
    definitions into rendered HTML fragments. Each method named
    ``block_<type>`` can be referenced by a YAML block with ``type: <type>``.

    Example:
        class MyBlocks(ReportBlockMixin):
            def block_my_summary(self, title: str = "My summary") -> str:
                return self._wrap_section_content(title, "<p>Custom content</p>")

        xpl.generate_report(..., block_class=MyBlocks)

    Runtime context
    ---------------
    During report generation, the runtime object exposes prepared attributes
    that custom blocks can use, such as:
    - `self.explainer`
    - `self.x_init` and `self.x_train_pre`
    - `self.df_train_test`
    - `self.y_train`, `self.y_test`, `self.y_pred`
    - `self.target_name`, `self.max_points`

    Design notes
    ------------
    - Each `block_*` method should return an HTML string.
    - Helper methods like `_wrap_section_content` and `_plotly_html` are
      provided to keep custom blocks concise and consistent with built-in ones.
    - Validation errors should raise `ValueError` with actionable messages.
    """

    def block_header(self, title: str = "Report", subtitle: str = "") -> str:
        """Render the report hero/header block.

        Parameters
        ----------
        title : str, default="Report"
            Main heading shown at the top of the report body.
        subtitle : str, default=""
            Optional descriptive text displayed in a callout below the title.

        Returns
        -------
        str
            HTML fragment containing the title and optional subtitle.
        """
        sub = f'<div class="shapash-callout"><p>{subtitle}</p></div>' if subtitle else ""
        return f'<div class="main-header"><h1>{title}</h1>{sub}</div>'

    def block_text(self, title: str = "", body: str = "", color: str = "gray") -> str:
        """Render a free-text content section.

        Parameters
        ----------
        title : str, default=""
            Optional section title.
        body : str, default=""
            Raw text or HTML-compatible content to display.
        color : str, default="gray"
            Reserved palette key for config parity. Not currently used directly
            by this block renderer.

        Returns
        -------
        str
            HTML fragment for a basic text paragraph section.
        """
        h2 = f'<h2 class="section-title">{title}</h2>' if title else ""
        return f'<div class="content-block">{h2}<p>{body}</p></div>'

    def block_project_information(
        self,
        title: str = "Project information",
        color: str = "gray",
        project_info_file: str = "",
        section_name: str | None = None,
    ) -> str:
        """Render a project-information section from a YAML metadata file.

        Parameters
        ----------
        title : str, default="Project information"
            Top-level section title.
        color : str, default="gray"
            Reserved palette key for config parity.
        project_info_file : str, default=""
            Path to a YAML file containing one or more mapping sections.
        section_name : str or None, default=None
            Optional section key to render. When omitted, all top-level mapping
            sections are rendered.

        Returns
        -------
        str
            HTML fragment containing one table per selected YAML section.

        Raises
        ------
        ValueError
            If ``project_info_file`` is missing, not found, invalid, or if
            ``section_name`` does not exist.
        """
        if not project_info_file:
            raise ValueError("project_information block requires the 'project_info_file' parameter.")

        config_path = Path(project_info_file).expanduser()
        if not config_path.is_absolute():
            config_path = Path.cwd() / config_path
        config_path = config_path.resolve()
        if not config_path.exists():
            raise ValueError(f"project_information file not found: {config_path}")

        with config_path.open(encoding="utf-8") as stream:
            project_info = yaml.safe_load(stream) or {}
        if not isinstance(project_info, dict):
            raise ValueError("project_information YAML must define a top-level mapping.")

        if section_name is not None:
            if section_name not in project_info:
                raise ValueError(f"Unknown project_information section: {section_name}")
            project_info = {section_name: project_info[section_name]}

        sections_html = []
        for current_section_name, section_values in project_info.items():
            if not isinstance(section_values, dict):
                continue
            rows = self._render_key_value_rows(section_values)
            sections_html.append(
                f'<div class="content-block"><h3 class="section-title" style="font-size:1.2rem">{current_section_name}</h3>'
                f'<table class="kv-table">{rows}</table></div>'
            )

        return self._wrap_section_content(title, "".join(sections_html))

    def block_key_value(self, title: str = "", items: dict | None = None, color: str = "gold") -> str:
        """Render a key-value table section.

        Parameters
        ----------
        title : str, default=""
            Optional section title.
        items : dict or None, default=None
            Mapping of label to value rows.
        color : str, default="gold"
            Reserved palette key for config parity.

        Returns
        -------
        str
            HTML fragment containing a two-column key-value table.
        """
        items = items or {}
        rows = self._render_key_value_rows(items)
        h2 = f'<h2 class="section-title">{title}</h2>' if title else ""
        return f'<div class="content-block">{h2}<table class="kv-table">{rows}</table></div>'

    def block_badge_row(self, title: str = "", badges: list | None = None) -> str:
        """Render a horizontal row of badge-style values.

        Parameters
        ----------
        title : str, default=""
            Optional section title.
        badges : list or None, default=None
            List of dictionaries describing badges. Supported keys include
            ``label``, ``value``, and ``color``.

        Returns
        -------
        str
            HTML fragment containing styled badges.
        """
        badges = badges or []
        pills = ""
        for badge in badges:
            palette = PALETTE.get(badge.get("color", "gray"), PALETTE["gray"])
            pills += (
                f'<span class="badge" style="border-color:{palette["border"]}">'
                f'<span style="color:{palette["title"]};font-weight:600">{badge.get("label", "")}</span>'
                f'<span style="margin-left:8px">{badge.get("value", "")}</span></span>'
            )
        h2 = f'<h2 class="section-title">{title}</h2>' if title else ""
        return f'<div class="content-block">{h2}<div style="display:flex;flex-wrap:wrap;gap:10px">{pills}</div></div>'

    def block_callout(self, body: str = "", color: str = "gold", icon: str = "") -> str:
        """Render a highlighted callout paragraph.

        Parameters
        ----------
        body : str, default=""
            Callout content.
        color : str, default="gold"
            Reserved palette key for config parity.
        icon : str, default=""
            Reserved icon identifier for future visual customization.

        Returns
        -------
        str
            HTML fragment for a callout container.
        """
        return f'<div class="shapash-callout"><p>{body}</p></div>'

    def block_divider(self, label: str = "") -> str:
        """Render a horizontal divider between sections.

        Parameters
        ----------
        label : str, default=""
            Reserved label value for future divider variants.

        Returns
        -------
        str
            HTML fragment for a divider line.
        """
        return '<div class="shapash-divider"></div>'

    def block_global_analysis(self, title: str = "", color: str = "gray") -> str:
        """Render a global train/test descriptive statistics table.

        Parameters
        ----------
        title : str, default=""
            Optional section title.
        color : str, default="gray"
            Reserved palette key for config parity.

        Returns
        -------
        str
            HTML fragment containing a statistics summary table.

        Raises
        ------
        ValueError
            If required train/test runtime data is unavailable.

        Notes
        -----
        When training data is available, both prediction and training dataset
        statistics are included side by side.
        """
        self._require_train_test_data("global_analysis")
        test_stats = perform_global_dataframe_analysis(self.x_init)
        train_stats = perform_global_dataframe_analysis(self.x_train_pre) if self.x_train_pre is not None else None
        stats_table = stats_to_table(
            test_stats=test_stats,
            train_stats=train_stats,
            names=["Prediction dataset", "Training dataset"],
        )
        table_html = stats_table.to_html(classes="kv-table", border=0)
        return self._wrap_section_content(title, table_html)

    def block_model_analysis(self, title: str = "Model information", color: str = "blue") -> str:
        """Render model metadata and parameter tables.

        Parameters
        ----------
        title : str, default="Model information"
            Section title.
        color : str, default="blue"
            Reserved palette key for config parity.

        Returns
        -------
        str
            HTML fragment including model type, library metadata, and parameter
            key/value tables.

        Raises
        ------
        ValueError
            If the runtime does not expose an explainer instance.
        """
        explainer = self._require_explainer("model_analysis")
        model = explainer.model

        model_module = model.__class__.__module__
        model_package = model_module.split(".")[0]
        package_name = "scikit-learn" if model_package == "sklearn" else model_package
        try:
            library_version = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            library_version = f"not found for {model_package}"

        model_params = getattr(model, "__dict__", {})
        params_items = list(model_params.items())
        split_idx = len(params_items) // 2

        def _truncate(value, max_len):
            text = str(value)
            return text if len(text) <= max_len else text[: max_len - 3] + "..."

        def _render_param_rows(items):
            return "".join(
                (
                    "<tr>"
                    f'<th scope="row" style="text-align:center">{_truncate(key, 50)}</th>'
                    f'<td style="text-align:center">{_truncate(val, 300)}</td>'
                    "</tr>"
                )
                for key, val in items
            )

        table_header = (
            "<thead><tr>"
            '<th scope="col" style="text-align:center">Parameter key</th>'
            '<th scope="col" style="text-align:center">Parameter value</th>'
            "</tr></thead>"
        )

        table_left = (
            '<table class="kv-table" style="table-layout:fixed">'
            f"{table_header}"
            f"<tbody>{_render_param_rows(params_items[:split_idx])}</tbody>"
            "</table>"
        )
        table_right = (
            '<table class="kv-table" style="table-layout:fixed">'
            f"{table_header}"
            f"<tbody>{_render_param_rows(params_items[split_idx:])}</tbody>"
            "</table>"
        )

        content = (
            '<div class="model-analysis-meta">'
            f'<p class="model-analysis-line"><strong>Model used :</strong> {model.__class__.__name__}</p>'
            f'<p class="model-analysis-line"><strong>Library :</strong> {model_module}</p>'
            f'<p class="model-analysis-line"><strong>Library version :</strong> {library_version}</p>'
            '<p class="model-analysis-line"><strong>Model parameters :</strong></p>'
            "</div>"
            '<div class="model-analysis-tables">'
            f'<div class="model-analysis-table-col">{table_left}</div>'
            f'<div class="model-analysis-table-col">{table_right}</div>'
            "</div>"
            "<hr>"
        )

        return self._wrap_section_content(title, content)

    def block_relationship_target(
        self,
        title: str = "Relationship with target variable",
        feature: str = "OverallQual",
        color: str = "blue",
        max_y: int | None = None,
    ) -> str:
        """Render a feature-vs-target box plot on training data.

        Parameters
        ----------
        title : str, default="Relationship with target variable"
            Section title.
        feature : str, default="OverallQual"
            Feature name from ``x_train_pre`` used on the x-axis.
        color : str, default="blue"
            Reserved palette key for config parity.
        max_y : int or None, default=None
            Optional upper bound for y-axis range.

        Returns
        -------
        str
            HTML fragment containing the rendered Plotly box chart.

        Raises
        ------
        ValueError
            If training data or target is missing, or if ``feature`` is unknown.
        """
        self._require_train_test_data("relationship_target")
        if self.x_train_pre is None or self.y_train is None:
            raise ValueError("relationship_target block requires both training features and y_train.")
        if feature not in self.x_train_pre.columns:
            raise ValueError(f"Unknown feature '{feature}' for relationship_target block.")

        target_name = self.target_name_train or "target"
        df_train = self.x_train_pre.copy()
        df_train[target_name] = self.y_train

        fig = px.box(df_train, x=feature, y=target_name)
        if max_y is not None:
            fig.update_yaxes(range=[0, max_y])
        return self._wrap_section_content(title, self._plotly_html(fig))

    def block_training_correlations(
        self,
        title: str = "Relationship between training variables",
        color: str = "blue",
        max_features: int = 30,
    ) -> str:
        """Render a training-only correlation heatmap.

        Parameters
        ----------
        title : str, default="Relationship between training variables"
            Section title.
        color : str, default="blue"
            Reserved palette key for config parity.
        max_features : int, default=30
            Maximum number of numeric features to keep on each heatmap axis.

        Returns
        -------
        str
            HTML fragment containing the correlation heatmap.

        Raises
        ------
        ValueError
            If ``x_train_pre`` is unavailable.
        """
        if self.x_train_pre is None:
            raise ValueError("training_correlations block requires x_train.")

        numeric_train = self.x_train_pre.select_dtypes(include="number")
        corr = numeric_train.corr(numeric_only=True)
        if max_features > 0 and corr.shape[0] > max_features:
            corr = corr.iloc[:max_features, :max_features]

        fig = px.imshow(corr, color_continuous_scale="YlGnBu", zmin=-1, zmax=1, aspect="auto")
        return self._wrap_section_content(title, self._plotly_html(fig))

    def block_performance_metrics(
        self,
        title: str = "Model performance",
        color: str = "orange",
        metrics: list | None = None,
    ) -> str:
        """Render configured performance metrics as badges.

        Parameters
        ----------
        title : str, default="Model performance"
            Section title.
        color : str, default="orange"
            Palette key used for badge accent color.
        metrics : list or None, default=None
            Metric configurations. Each item should contain ``path`` and may
            define ``name``.

        Returns
        -------
        str
            HTML fragment containing one badge per computed metric.

        Raises
        ------
        ValueError
            If ``y_test`` or ``y_pred`` is missing.
        """
        if self.y_test is None or self.y_pred is None:
            raise ValueError("performance_metrics block requires y_test and y_pred.")

        metric_items = []
        metrics = metrics or []
        for metric_cfg in metrics:
            metric_path = metric_cfg.get("path")
            metric_name = metric_cfg.get("name", metric_path)
            if not metric_path:
                continue
            module_path, fn_name = metric_path.rsplit(".", 1)
            metric_fn = getattr(importlib.import_module(module_path), fn_name)
            value = metric_fn(self.y_test, self.y_pred)
            metric_items.append({"label": metric_name, "value": f"{value:,.2f}", "color": color})

        return self.block_badge_row(title=title, badges=metric_items)

    def block_pred_vs_true(self, title: str = "y_pred vs y_test", color: str = "orange") -> str:
        """Render a scatter plot comparing predictions to true values.

        Parameters
        ----------
        title : str, default="y_pred vs y_test"
            Section title.
        color : str, default="orange"
            Reserved palette key for config parity.

        Returns
        -------
        str
            HTML fragment containing a Plotly scatter plot.

        Raises
        ------
        ValueError
            If ``y_test`` or ``y_pred`` is missing.
        """
        if self.y_test is None or self.y_pred is None:
            raise ValueError("pred_vs_true block requires y_test and y_pred.")

        scatter_df = pd.DataFrame({"y_test": self.y_test, "y_pred": self.y_pred})
        fig = px.scatter(scatter_df, x="y_test", y="y_pred")
        return self._wrap_section_content(title, self._plotly_html(fig))

    def block_feature_distribution(
        self,
        feature: str,
        title: str = "",
        color: str = "blue",
        dataset_split: str = "data_train_test",
        prediction_label: str = "test",
        training_label: str = "train",
        width: int = 700,
        height: int = 500,
    ) -> str:
        """Render a feature distribution plot across dataset splits.

        Parameters
        ----------
        feature : str
            Feature column to analyze.
        title : str, default=""
            Optional section title. If empty, uses feature label.
        color : str, default="blue"
            Reserved palette key for config parity.
        dataset_split : str, default="data_train_test"
            Name of the split column in ``df_train_test`` used for hue.
        prediction_label : str, default="test"
            Reserved label for prediction split compatibility.
        training_label : str, default="train"
            Reserved label for training split compatibility.
        width : int, default=700
            Plot width in pixels.
        height : int, default=500
            Plot height in pixels.

        Returns
        -------
        str
            HTML fragment containing the rendered feature distribution plot.

        Raises
        ------
        ValueError
            If train/test runtime data is unavailable or ``feature`` is missing.
        """
        self._require_train_test_data("feature_distribution")
        if feature not in self.df_train_test.columns:
            raise ValueError(f"Unknown feature '{feature}' for feature_distribution block.")

        fig = plot_distribution(
            df_all=self.df_train_test,
            col=feature,
            hue=dataset_split,
            colors_dict=self._feature_distribution_colors(),
            width=width,
            height=height,
        )
        return self._wrap_section_content(title or self._feature_label(feature), self._plotly_html(fig))

    def block_correlations_plot(
        self,
        title: str = "",
        color: str = "blue",
        max_features: int = 20,
        width: int | None = None,
        height: int = 500,
    ) -> str:
        """Render the explainer correlation plot.

        Parameters
        ----------
        title : str, default=""
            Optional section title.
        color : str, default="blue"
            Reserved palette key for config parity.
        max_features : int, default=20
            Maximum number of features included in the correlation matrix.
        width : int or None, default=None
            Plot width in pixels. If ``None``, width is inferred from the number
            of dataset splits.
        height : int, default=500
            Plot height in pixels.

        Returns
        -------
        str
            HTML fragment containing the rendered correlation chart.

        Raises
        ------
        ValueError
            If required explainer or train/test data is unavailable.

        Notes
        -----
        When both training and prediction datasets are available, the plot is
        faceted by dataset split.
        """
        self._require_train_test_data("correlations_plot")
        explainer = self._require_explainer("correlations_plot")
        resolved_width = width or (900 if len(self.df_train_test["data_train_test"].unique()) > 1 else 500)
        fig = explainer.plot.correlations_plot(
            self.df_train_test,
            optimized=True,
            facet_col="data_train_test",
            max_features=max_features,
            width=resolved_width,
            height=height,
        )
        return self._wrap_section_content(title, self._plotly_html(fig))

    def block_feature_importance(self, title: str = "", color: str = "green", label=None) -> str:
        """Render the explainer feature-importance plot.

        Parameters
        ----------
        title : str, default=""
            Optional section title.
        color : str, default="green"
            Reserved palette key for config parity.
        label : object, optional
            Optional class label used in classification contexts.

        Returns
        -------
        str
            HTML fragment containing the feature-importance chart.

        Raises
        ------
        ValueError
            If the runtime does not expose an explainer instance.
        """
        explainer = self._require_explainer("feature_importance")
        fig = explainer.plot.features_importance(label=label)
        return self._wrap_section_content(title, self._plotly_html(fig))

    def block_contribution_plot(
        self,
        feature: str | None = None,
        title: str = "",
        color: str = "green",
        label=None,
        max_points: int | None = None,
        include_all_features: bool = False,
        group_id: str = "contribution",
    ) -> str:
        """Render contribution plot(s) for one feature or all features.

        Parameters
        ----------
        feature : str or None, default=None
            Feature to plot when ``include_all_features`` is ``False``.
        title : str, default=""
            Optional section title.
        color : str, default="green"
            Reserved palette key for config parity.
        label : object, optional
            Optional class label used in classification contexts.
        max_points : int or None, default=None
            Maximum number of points for plot sampling. Falls back to
            ``self.max_points`` when omitted.
        include_all_features : bool, default=False
            If ``True``, renders a feature selector and one panel per feature.
        group_id : str, default="contribution"
            HTML prefix used to namespace selector and panel identifiers.

        Returns
        -------
        str
            HTML fragment containing one contribution plot or an interactive
            selector with multiple contribution plots.

        Raises
        ------
        ValueError
            If required explainer data is unavailable, or if ``feature`` is
            missing when ``include_all_features`` is ``False``.
        """
        explainer = self._require_explainer("contribution_plot")

        if not include_all_features:
            if feature is None:
                raise ValueError("contribution_plot block requires 'feature' when include_all_features=False.")
            fig = explainer.plot.contribution_plot(feature, label=label, max_points=max_points or self.max_points)
            for trace in fig.data:
                if trace.type == "bar":
                    trace.marker.color = "lightgrey"
            return self._wrap_section_content(title or self._feature_label(feature), self._plotly_html(fig))

        if getattr(explainer, "x_init", None) is None:
            raise ValueError("contribution_plot block with include_all_features=True requires explainer.x_init.")

        feature_names = list(explainer.x_init.columns)
        if not feature_names:
            return self._wrap_section_content(title, '<div class="content-block"><p>No feature available.</p></div>')

        sorted_features = sorted(
            feature_names,
            key=lambda current_feature: (str(self._feature_label(current_feature)).lower(), str(current_feature)),
        )

        instance_id = uuid4().hex[:8]
        selector_id = f"{group_id}-selector-{instance_id}"
        feature_panels = []
        feature_options = []

        for idx, feature_name in enumerate(sorted_features):
            fig = explainer.plot.contribution_plot(feature_name, label=label, max_points=max_points or self.max_points)
            for trace in fig.data:
                if trace.type == "bar":
                    trace.marker.color = "lightgrey"

            feature_id = f"{group_id}-feature-{idx}-{instance_id}"
            feature_label = self._feature_label(feature_name)
            feature_options.append(f'<option value="{feature_id}">{feature_label}</option>')
            feature_panels.append(
                f'<div id="{feature_id}" class="section-block contribution-feature-panel" '
                f'data-panel-group="{selector_id}" style="display:none">'
                f"{self._plotly_html(fig)}"
                "</div>"
            )

        controls_html = (
            '<div class="univariate-picker">'
            f'<label for="{selector_id}">Choose a feature</label>'
            f'<select id="{selector_id}" class="univariate-select js-panel-select" data-panel-group="{selector_id}">{"".join(feature_options)}</select>'
            '</div>'
        )

        resolved_title = title or "Features contribution plots"
        return self._wrap_section_content(resolved_title, f'{controls_html}{"".join(feature_panels)}')

    def block_interactions_plot(
        self,
        title: str = "",
        color: str = "green",
        col1: str | None = None,
        col2: str | None = None,
        max_points: int | None = None,
    ) -> str:
        """Render a two-feature interaction plot.

        Parameters
        ----------
        title : str, default=""
            Optional section title. If empty, uses resolved feature labels.
        color : str, default="green"
            Reserved palette key for config parity.
        col1 : str or None, default=None
            First feature name. If omitted with ``col2``, an interaction pair is
            auto-selected.
        col2 : str or None, default=None
            Second feature name.
        max_points : int or None, default=None
            Maximum number of points for plot sampling.

        Returns
        -------
        str
            HTML fragment containing the interaction chart.

        Raises
        ------
        ValueError
            If no valid interaction pair can be resolved.
        """
        explainer = self._require_explainer("interactions_plot")
        feature_one, feature_two = self._resolve_interaction_pair(col1, col2)
        fig = explainer.plot.interactions_plot(
            col1=feature_one, col2=feature_two, max_points=max_points or self.max_points
        )
        resolved_title = title or f"{self._feature_label(feature_one)} / {self._feature_label(feature_two)}"
        return self._wrap_section_content(resolved_title, self._plotly_html(fig))

    def block_target_distribution(
        self,
        title: str = "",
        color: str = "blue",
        width: int = 700,
        height: int = 500,
    ) -> str:
        """Render predicted-vs-true target distributions.

        Parameters
        ----------
        title : str, default=""
            Optional section title.
        color : str, default="blue"
            Reserved palette key for config parity.
        width : int, default=700
            Plot width in pixels.
        height : int, default=500
            Plot height in pixels.

        Returns
        -------
        str
            HTML fragment containing the overlaid target distributions.

        Raises
        ------
        ValueError
            If target or prediction values are missing.
        """
        self._require_explainer("target_distribution")
        if self.y_test is None or self.y_pred is None:
            raise ValueError("target_distribution block requires y_test and predicted values from the explainer.")

        target_name = self.target_name or "target"
        df_target = pd.concat(
            [
                pd.DataFrame({target_name: self.y_pred}).assign(_dataset="pred"),
                pd.DataFrame({target_name: self.y_test}).assign(_dataset="true"),
            ]
        ).reset_index(drop=True)
        fig = plot_distribution(
            df_all=df_target,
            col=target_name,
            hue="_dataset",
            colors_dict=self._performance_distribution_colors(),
            width=width,
            height=height,
        )
        return self._wrap_section_content(title or "Target distribution", self._plotly_html(fig))

    def block_target_analysis(
        self,
        title: str = "Target analysis",
        show_train: bool = True,
        width: int = 700,
        height: int = 500,
    ) -> str:
        """Render a univariate analysis focused on the target variable.

        Parameters
        ----------
        title : str, default="Target analysis"
            Section title.
        show_train : bool, default=True
            Whether to include training-target distribution and stats when
            ``y_train`` is available.
        width : int, default=700
            Plot width in pixels.
        height : int, default=500
            Plot height in pixels.

        Returns
        -------
        str
            HTML fragment combining summary statistics and a distribution chart.

        Raises
        ------
        ValueError
            If ``y_test`` is missing.
        """
        from shapash.report.common import compute_col_types, series_dtype
        from shapash.report.data_analysis import perform_univariate_dataframe_analysis

        if self.y_test is None:
            raise ValueError("target_analysis block requires y_test.")

        target_name = self.target_name or "target"
        y_test_series = pd.Series(self.y_test, name=target_name)
        y_train_series = pd.Series(self.y_train, name=target_name) if self.y_train is not None and show_train else None

        analysis_source = pd.DataFrame({target_name: y_test_series})
        if y_train_series is not None:
            analysis_source = pd.concat(
                [analysis_source, pd.DataFrame({target_name: y_train_series})], ignore_index=True
            )

        col_types = compute_col_types(analysis_source)
        test_stats = perform_univariate_dataframe_analysis(
            pd.DataFrame({target_name: y_test_series}), col_types=col_types
        )
        train_stats = (
            perform_univariate_dataframe_analysis(pd.DataFrame({target_name: y_train_series}), col_types=col_types)
            if y_train_series is not None
            else None
        )

        names = ["Prediction dataset", "Training dataset"]
        target_stats = stats_to_table(
            test_stats=test_stats[target_name],
            train_stats=train_stats[target_name] if train_stats is not None else None,
            names=names,
        )

        distribution_df = pd.concat(
            [
                pd.DataFrame({target_name: y_test_series}).assign(data_train_test="test"),
                pd.DataFrame({target_name: y_train_series}).assign(data_train_test="train")
                if y_train_series is not None
                else pd.DataFrame(columns=[target_name, "data_train_test"]),
            ],
            ignore_index=True,
        )
        fig = plot_distribution(
            df_all=distribution_df,
            col=target_name,
            hue="data_train_test",
            colors_dict=self._feature_distribution_colors(),
            width=width,
            height=height,
        )

        dtype_label = str(series_dtype(y_test_series))
        target_header = (
            '<div class="content-block">'
            f"<p><strong>{target_name}</strong> "
            f'<span style="font-weight:400;color:var(--text-light);font-size:12px">({dtype_label})</span></p>'
            "</div>"
        )
        panel_html = (
            '<div class="section-block univariate-feature-panel">'
            '<div class="analysis-side-by-side">'
            f'<div class="analysis-side-table">{target_stats.to_html(classes="kv-table", border=0)}</div>'
            f'<div class="analysis-side-plot">{self._plotly_html(fig)}</div>'
            '</div>'
            '</div>'
        )

        return self._wrap_section_content(title, f"{target_header}{panel_html}")

    def block_confusion_matrix(self, title: str = "", color: str = "orange") -> str:
        """Render a classification confusion matrix.

        Parameters
        ----------
        title : str, default=""
            Optional section title.
        color : str, default="orange"
            Reserved palette key for config parity.

        Returns
        -------
        str
            HTML fragment containing the confusion-matrix chart.

        Raises
        ------
        ValueError
            If the explainer, ``y_test``, or ``y_pred`` is unavailable.
        """
        explainer = self._require_explainer("confusion_matrix")
        if self.y_test is None or self.y_pred is None:
            raise ValueError("confusion_matrix block requires y_test and predicted values from the explainer.")
        fig = plot_confusion_matrix(y_true=self.y_test, y_pred=self.y_pred, colors_dict=explainer.colors_dict)
        return self._wrap_section_content(title or "Confusion matrix", self._plotly_html(fig))

    def block_univariate_analysis(
        self,
        title: str = "Univariate analysis",
        show_train: bool = True,
        group_id: str = "univariate",
    ) -> str:
        """Render a univariate analysis for all explainer features.

        For each feature, renders a distribution plot and a summary statistics
        table. When training data is available and show_train is True, statistics
        are shown for both prediction and training datasets side by side.

        Parameters
        ----------
        title : str
            Section title displayed above the analysis.
        show_train : bool
            Whether to include training data alongside prediction data.
        group_id : str
            HTML identifier prefix used to namespace the dropdown and feature panels.

        Returns
        -------
        str
            HTML fragment with a feature selector and one analysis panel per
            available feature.

        Raises
        ------
        ValueError
            If required explainer or train/test data is unavailable.
        """
        from shapash.report.common import compute_col_types, series_dtype
        from shapash.report.data_analysis import perform_univariate_dataframe_analysis

        self._require_train_test_data("univariate_analysis")
        explainer = self._require_explainer("univariate_analysis")

        df = self.df_train_test
        col_splitter = "data_train_test"
        names = ["Prediction dataset", "Training dataset"]

        col_types = compute_col_types(df)
        n_splits = df[col_splitter].nunique()

        test_stats = perform_univariate_dataframe_analysis(df.loc[df[col_splitter] == "test"], col_types=col_types)
        train_stats = (
            perform_univariate_dataframe_analysis(df.loc[df[col_splitter] == "train"], col_types=col_types)
            if n_splits > 1 and show_train
            else None
        )

        list_cols_labels = sorted(
            explainer.features_dict.get(col, col) for col in df.drop(col_splitter, axis=1).columns
        )

        feature_panels = []
        feature_options = []
        instance_id = uuid4().hex[:8]
        selector_id = f"{group_id}-selector-{instance_id}"

        for idx, col_label in enumerate(list_cols_labels):
            col = explainer.inv_features_dict.get(col_label, col_label)
            if col not in test_stats:
                continue

            fig = plot_distribution(
                df_all=df,
                col=col,
                hue=col_splitter,
                colors_dict=self._feature_distribution_colors(),
            )
            col_stats = stats_to_table(
                test_stats=test_stats[col],
                train_stats=train_stats[col] if train_stats is not None else None,
                names=names,
            )

            feature_id = f"{group_id}-feature-{idx}-{instance_id}"
            dtype_label = str(series_dtype(df[col]))

            feature_options.append(f'<option value="{feature_id}">{col_label} ({dtype_label})</option>')
            feature_panels.append(
                f'<div id="{feature_id}" class="section-block univariate-feature-panel" '
                f'data-panel-group="{selector_id}" style="display:none">'
                '<div class="analysis-side-by-side">'
                f'<div class="analysis-side-table">{col_stats.to_html(classes="kv-table", border=0)}</div>'
                f'<div class="analysis-side-plot">{self._plotly_html(fig)}</div>'
                '</div>'
                f'</div>'
            )

        if not feature_panels:
            return self._wrap_section_content(title, '<div class="content-block"><p>No feature available.</p></div>')

        controls_html = (
            '<div class="univariate-picker">'
            f'<label for="{selector_id}">Choose a feature</label>'
            f'<select id="{selector_id}" class="univariate-select js-panel-select" data-panel-group="{selector_id}">{"".join(feature_options)}</select>'
            '</div>'
        )

        return self._wrap_section_content(title, f'{controls_html}{"".join(feature_panels)}')

    def _preprocess_train_data(self, x_train: pd.DataFrame | None) -> pd.DataFrame | None:
        """Apply inverse preprocessing and postprocessing to training data.

        Parameters
        ----------
        x_train : pandas.DataFrame or None
            Raw training frame in encoded/model-input space.

        Returns
        -------
        pandas.DataFrame or None
            Prepared training frame aligned with ``x_init`` representation.
        """
        if x_train is None or self.explainer is None:
            return x_train
        x_train_pre = inverse_transform(x_train, self.explainer.preprocessing)
        x_train_pre = handle_categorical_missing(x_train_pre)
        if self.explainer.postprocessing:
            x_train_pre = apply_postprocessing(x_train_pre, self.explainer.postprocessing)
        return x_train_pre

    @staticmethod
    def _get_values_and_name(y: pd.DataFrame | pd.Series | list | None, default_name: str) -> tuple[object, str | None]:
        """Normalize target-like input to value array/list and display name.

        Parameters
        ----------
        y : pandas.DataFrame, pandas.Series, list, or None
            Input target values.
        default_name : str
            Fallback display name when ``y`` is provided as a list.

        Returns
        -------
        tuple[object, str or None]
            Pair ``(values, name)`` where values is an array-like object.

        Raises
        ------
        ValueError
            If DataFrame input has more than one column or if type is unsupported.
        """
        if y is None:
            return None, None
        if isinstance(y, pd.DataFrame):
            if len(y.columns) != 1:
                raise ValueError("Number of columns found is greater than 1")
            return y.values[:, 0], y.columns[0]
        if isinstance(y, pd.Series):
            return y.values, y.name
        if isinstance(y, list):
            return y, default_name
        raise ValueError(f"Cannot process following type : {type(y)}")

    @staticmethod
    def _create_train_test_df(test: pd.DataFrame | None, train: pd.DataFrame | None) -> pd.DataFrame | None:
        """Build a combined train/test frame with split markers.

        Parameters
        ----------
        test : pandas.DataFrame or None
            Prediction/test dataset.
        train : pandas.DataFrame or None
            Training dataset.

        Returns
        -------
        pandas.DataFrame or None
            Concatenated frame with ``data_train_test`` marker column, or
            ``None`` when both inputs are ``None``.

        Raises
        ------
        ValueError
            If input frames already contain reserved ``data_train_test`` column.
        """
        if (test is not None and "data_train_test" in test.columns) or (
            train is not None and "data_train_test" in train.columns
        ):
            raise ValueError('"data_train_test" column must be renamed as it is reserved by smart report runtime')
        if test is None and train is None:
            return None
        frames = []
        if test is not None:
            frames.append(test.assign(data_train_test="test"))
        if train is not None:
            frames.append(train.assign(data_train_test="train"))
        return pd.concat(frames).reset_index(drop=True)

    def _require_explainer(self, block_type: str):
        """Ensure an explainer is attached to the runtime.

        Parameters
        ----------
        block_type : str
            Calling block name for contextual error messaging.

        Returns
        -------
        object
            Runtime explainer.

        Raises
        ------
        ValueError
            If explainer is missing.
        """
        if self.explainer is None:
            raise ValueError(f"{block_type} block requires an explainer on the report instance.")
        return self.explainer

    def _require_train_test_data(self, block_type: str) -> None:
        """Ensure combined train/test data is available on the runtime.

        Parameters
        ----------
        block_type : str
            Calling block name for contextual error messaging.

        Raises
        ------
        ValueError
            If combined train/test dataframe is missing.
        """
        if self.df_train_test is None:
            raise ValueError(f"{block_type} block requires x_train and explainer.x_init data on the report instance.")

    def _resolve_interaction_pair(self, col1: str | None, col2: str | None) -> tuple[str, str]:
        """Resolve an interaction pair either explicitly or automatically.

        Parameters
        ----------
        col1 : str or None
            First feature name.
        col2 : str or None
            Second feature name.

        Returns
        -------
        tuple[str, str]
            Pair of resolved feature names.

        Raises
        ------
        ValueError
            If automatic selection cannot find any interaction pair.
        """
        if col1 and col2:
            return col1, col2
        explainer = self._require_explainer("interactions_plot")
        list_ind, _ = explainer.plot._select_indices_interactions_plot(selection=None, max_points=self.max_points)
        interaction_values = explainer.get_interaction_values(selection=list_ind)
        sorted_indices = compute_sorted_variables_interactions_list_indices(interaction_values)
        if not sorted_indices:
            raise ValueError("No interaction pair available for interactions_plot block.")
        first_idx, second_idx = sorted_indices[0]
        return explainer.columns_dict[first_idx], explainer.columns_dict[second_idx]

    def _feature_label(self, feature: str) -> str:
        """Return display label for a technical feature name.

        Parameters
        ----------
        feature : str
            Technical feature identifier.

        Returns
        -------
        str
            Business/display feature label when available, otherwise ``feature``.
        """
        if self.explainer is None:
            return feature
        return self.explainer.features_dict.get(feature, feature)

    def _feature_distribution_colors(self) -> dict:
        """Return color mapping used by distribution blocks.

        Returns
        -------
        dict
            Color palette mapping from report style configuration.
        """
        explainer = self._require_explainer("feature_distribution")
        return explainer.colors_dict["report_feature_distribution"]

    @staticmethod
    def _performance_distribution_colors() -> dict:
        """Return default color mapping for predicted-vs-true target plots.

        Returns
        -------
        dict
            Static mapping with ``pred`` and ``true`` color values.
        """
        return {"pred": "#2255aa", "true": "#f4c000"}

    @staticmethod
    def _plotly_html(fig) -> str:
        """Convert a Plotly figure into embeddable HTML.

        Parameters
        ----------
        fig : plotly.graph_objects.Figure
            Plotly figure instance.

        Returns
        -------
        str
            HTML fragment that can be inserted into report blocks.
        """
        return render_plotly_pane_html(fig)

    @staticmethod
    def _render_key_value_rows(items: dict) -> str:
        """Render table rows for a key-value mapping.

        Parameters
        ----------
        items : dict
            Mapping of key to display value.

        Returns
        -------
        str
            HTML table-row fragment.
        """
        return "".join(
            f'<tr><td class="kv-key"><span class="kv-key-label">{key}</span><span class="kv-key-sep"> :</span></td>'
            f'<td class="kv-val">{value}</td></tr>'
            for key, value in items.items()
        )

    @staticmethod
    def _wrap_section_content(title: str, body_html: str) -> str:
        """Wrap block body content in a standard report section container.

        Parameters
        ----------
        title : str
            Section title. If empty, no heading is rendered.
        body_html : str
            HTML fragment representing block content.

        Returns
        -------
        str
            Full HTML section fragment.
        """
        parts = []
        if title:
            parts.append(f'<h2 class="section-title">{title}</h2>')
        parts.append(body_html)
        return f'<div class="section-block">{"".join(parts)}</div>'
