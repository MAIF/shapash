"""Block implementations and report data helpers for smart reports."""

from __future__ import annotations

import importlib
import importlib.metadata
from pathlib import Path
from typing import Any

import pandas as pd
import panel as pn
import yaml

from shapash.plots.plot_evaluation_metrics import plot_confusion_matrix
from shapash.plots.plot_univariate import plot_distribution
from shapash.report.common import compute_col_types, series_dtype
from shapash.report.data_analysis import perform_global_dataframe_analysis, perform_univariate_dataframe_analysis
from shapash.report.panel_support import make_plotly_pane
from shapash.report.validation import stats_to_table
from shapash.utils.transform import apply_postprocessing, handle_categorical_missing, inverse_transform
from shapash.utils.utils import compute_sorted_variables_interactions_list_indices

PALETTE = {
    "gold": {"bg": "#ffffff", "border": "#f4c000", "title": "#f4c000", "text": "#343736"},
    "blue": {"bg": "#ffffff", "border": "#2255aa", "title": "#2255aa", "text": "#343736"},
    "gray": {"bg": "#ffffff", "border": "#eeeeee", "title": "#666666", "text": "#666666"},
    "orange": {"bg": "#fff9e6", "border": "#f4c000", "title": "#cc8833", "text": "#444444"},
}


class ReportBlockMixin:
    """Base mixin providing built-in and user-extensible smart report blocks."""

    def block_header(self, title: str = "Report", subtitle: str = "") -> pn.Column:
        """Render the report header section.

        Parameters
        ----------
        title : str, default="Report"
            Main report title displayed as a first-level heading.
        subtitle : str, default=""
            Optional markdown text displayed below the title.

        Returns
        -------
        pn.Column
            Panel column containing the title and optional subtitle.

        Examples
        --------
        >>> runtime.block_header(title="Model report", subtitle="Summary for Q2")
        """
        blocks: list[pn.viewable.Viewable] = [pn.pane.Markdown(f"# {title}", css_classes=["main-header"])]
        if subtitle:
            blocks.append(
                pn.pane.Markdown(
                    subtitle,
                    css_classes=["shapash-callout"],
                )
            )
        return pn.Column(*blocks, sizing_mode="stretch_width")

    def block_text(self, title: str = "", body: str = "") -> pn.Column:
        """Render a markdown text section.

        Parameters
        ----------
        title : str, default=""
            Optional section title.
        body : str, default=""
            Markdown content displayed in the block body.

        Returns
        -------
        pn.Column
            Panel column containing the wrapped text section.

        Examples
        --------
        >>> runtime.block_text(title="Context", body="This report compares train and test data.")
        """
        content: list[pn.viewable.Viewable] = []
        if body:
            content.append(pn.pane.Markdown(body, css_classes=["content-block"]))
        return self._wrap_section_content(title, content)

    def block_project_information(
        self,
        title: str = "Project information",
        project_info_file: str = "",
        section_name: str | None = None,
    ) -> pn.Column:
        """Render project metadata from a YAML configuration file.

        Parameters
        ----------
        title : str, default="Project information"
            Section title displayed above the metadata cards.
        project_info_file : str, default=""
            Path to a YAML file containing top-level mapping sections.
        section_name : str or None, default=None
            Optional section key to render only one subsection from the YAML file.

        Returns
        -------
        pn.Column
            Panel column containing one or more key-value tables.

        Examples
        --------
        >>> runtime.block_project_information(project_info_file="project_info.yaml")
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

        blocks: list[pn.viewable.Viewable] = []
        for current_section_name, section_values in project_info.items():
            if not isinstance(section_values, dict):
                continue
            df = pd.DataFrame(
                {"Key": list(section_values.keys()), "Value": [str(value) for value in section_values.values()]}
            )
            blocks.append(
                pn.Column(
                    pn.pane.Markdown(f"### {current_section_name}"),
                    pn.pane.DataFrame(df, index=False, sizing_mode="stretch_width", css_classes=["kv-table"]),
                    css_classes=["content-block", "project-info-card"],
                    sizing_mode="stretch_width",
                )
            )

        if not blocks:
            blocks = [pn.pane.Markdown("No project information available.", css_classes=["content-block"])]

        project_info_grid = pn.Column(
            *blocks,
            css_classes=["project-info-grid"],
            sizing_mode="stretch_width",
        )

        return self._wrap_section_content(title, [project_info_grid])

    def block_badge_row(self, title: str = "", badges: list | None = None) -> pn.Column:
        """Render a row of summary badges.

        Parameters
        ----------
        title : str, default=""
            Optional section title.
        badges : list or None, default=None
            List of dictionaries with keys such as ``label``, ``value``, and ``color``.

        Returns
        -------
        pn.Column
            Panel column containing a horizontal row of badge elements.

        Examples
        --------
        >>> runtime.block_badge_row(badges=[{"label": "AUC", "value": "0.89", "color": "blue"}])
        """
        badges = badges or []
        pills: list[pn.viewable.Viewable] = []
        for badge in badges:
            color_name = badge.get("color", "gray")
            if color_name not in PALETTE:
                color_name = "gray"
            pills.append(
                pn.pane.Markdown(
                    f"**{badge.get('label', '')}**: {badge.get('value', '')}",
                    css_classes=["badge-pill", f"badge-pill-{color_name}"],
                )
            )

        return self._wrap_section_content(title, [pn.Row(*pills, sizing_mode="stretch_width")])

    def block_callout(self, body: str = "") -> pn.Column:
        """Render a highlighted callout message.

        Parameters
        ----------
        body : str, default=""
            Markdown message to emphasize in the report.

        Returns
        -------
        pn.Column
            Panel column containing a styled callout pane.

        Examples
        --------
        >>> runtime.block_callout(body="Use this report for decision support only.")
        """
        return pn.Column(
            pn.pane.Markdown(
                body,
                css_classes=["shapash-callout"],
            ),
            sizing_mode="stretch_width",
        )

    def block_global_analysis(self, title: str = "") -> pn.Column:
        """Render global summary statistics for prediction and training datasets.

        Parameters
        ----------
        title : str, default=""
            Optional section title.

        Returns
        -------
        pn.Column
            Panel column containing a global statistics comparison table.

        Examples
        --------
        >>> runtime.block_global_analysis(title="Global dataset comparison")
        """
        self._require_train_test_data("global_analysis")
        test_stats = perform_global_dataframe_analysis(self.x_init)
        train_stats = perform_global_dataframe_analysis(self.x_train_pre) if self.x_train_pre is not None else None
        stats_table = stats_to_table(
            test_stats=test_stats,
            train_stats=train_stats,
            names=["Prediction dataset", "Training dataset"],
        )
        return self._wrap_section_content(
            title,
            [pn.pane.DataFrame(stats_table, sizing_mode="stretch_width", css_classes=["kv-table"])],
        )

    def block_model_analysis(self, title: str = "Model information") -> pn.Column:
        """Render model metadata and parameter tables.

        Parameters
        ----------
        title : str, default="Model information"
            Section title displayed above model details.

        Returns
        -------
        pn.Column
            Panel column containing model identity details and parameter tables.

        Examples
        --------
        >>> runtime.block_model_analysis()
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

        def _truncate(value: Any, max_len: int) -> str:
            text = str(value)
            return text if len(text) <= max_len else text[: max_len - 3] + "..."

        left_df = pd.DataFrame(
            {
                "Parameter key": [_truncate(key, 50) for key, _ in params_items[:split_idx]],
                "Parameter value": [_truncate(val, 300) for _, val in params_items[:split_idx]],
            }
        )
        right_df = pd.DataFrame(
            {
                "Parameter key": [_truncate(key, 50) for key, _ in params_items[split_idx:]],
                "Parameter value": [_truncate(val, 300) for _, val in params_items[split_idx:]],
            }
        )

        content: list[pn.viewable.Viewable] = [
            pn.pane.Markdown(
                "\n".join(
                    [
                        f"**Model used**: {model.__class__.__name__}",
                        f"**Library**: {model_module}",
                        f"**Library version**: {library_version}",
                        "**Model parameters**",
                    ]
                )
            ),
            pn.Row(
                pn.pane.DataFrame(left_df, sizing_mode="stretch_width", css_classes=["kv-table"]),
                pn.Spacer(width=24),
                pn.pane.DataFrame(right_df, sizing_mode="stretch_width", css_classes=["kv-table"]),
                sizing_mode="stretch_width",
            ),
        ]

        return self._wrap_section_content(title, content)

    def block_performance_metrics(
        self,
        title: str = "Model performance",
        color: str = "orange",
        metrics: list | None = None,
    ) -> pn.Column:
        """Compute and render selected evaluation metrics as badges.

        Parameters
        ----------
        title : str, default="Model performance"
            Section title displayed above metric badges.
        color : str, default="orange"
            Badge color name used for rendered metric pills.
        metrics : list or None, default=None
            Metric specifications with import path and optional display name.

        Returns
        -------
        pn.Column
            Panel column containing computed metric badges.

        Examples
        --------
        >>> runtime.block_performance_metrics(metrics=[{"path": "sklearn.metrics.accuracy_score"}])
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

    def block_feature_distribution(
        self,
        feature: str,
        title: str = "",
        dataset_split: str = "data_train_test",
        width: int = 700,
        height: int = 500,
    ) -> pn.Column:
        """Render feature distribution by dataset split.

        Parameters
        ----------
        feature : str
            Feature name to visualize.
        title : str, default=""
            Optional custom section title.
        dataset_split : str, default="data_train_test"
            Column used as hue to separate train/test distributions.
        width : int, default=700
            Plot width in pixels.
        height : int, default=500
            Plot height in pixels.

        Returns
        -------
        pn.Column
            Panel column containing the feature distribution plot.

        Examples
        --------
        >>> runtime.block_feature_distribution(feature="age")
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
        return self._wrap_section_content(title or self._feature_label(feature), [self._plotly_pane(fig)])

    def block_correlations_plot(
        self,
        title: str = "",
        max_features: int = 20,
        width: int | None = None,
        height: int = 500,
    ) -> pn.Column:
        """Render a feature correlation matrix.

        Parameters
        ----------
        title : str, default=""
            Optional section title.
        max_features : int, default=20
            Maximum number of features included in the matrix.
        width : int or None, default=None
            Optional explicit plot width.
        height : int, default=500
            Plot height in pixels.

        Returns
        -------
        pn.Column
            Panel column containing the correlations plot.

        Examples
        --------
        >>> runtime.block_correlations_plot(max_features=15)
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
        return self._wrap_section_content(title, [self._plotly_pane(fig)])

    def block_feature_importance(self, title: str = "", label=None) -> pn.Column:
        """Render global feature importance.

        Parameters
        ----------
        title : str, default=""
            Optional section title.
        label : Any, default=None
            Optional class/target label for label-specific importance.

        Returns
        -------
        pn.Column
            Panel column containing the feature-importance figure.

        Examples
        --------
        >>> runtime.block_feature_importance()
        """
        explainer = self._require_explainer("feature_importance")
        fig = explainer.plot.features_importance(label=label)
        return self._wrap_section_content(title, [self._plotly_pane(fig)])

    def block_contribution_plot(
        self,
        feature: str | None = None,
        title: str = "",
        label=None,
        max_points: int | None = None,
        include_all_features: bool = False,
    ) -> pn.Column:
        """Render feature contribution plots.

        Parameters
        ----------
        feature : str or None, default=None
            Feature name for single-feature mode.
        title : str, default=""
            Optional section title.
        label : Any, default=None
            Optional class/target label.
        max_points : int or None, default=None
            Maximum number of points used by the plotting backend.
        include_all_features : bool, default=False
            If True, create an interactive selector over contribution plots for all features.

        Returns
        -------
        pn.Column
            Panel column containing one contribution plot or selector-driven plots.

        Examples
        --------
        >>> runtime.block_contribution_plot(feature="age")
        >>> runtime.block_contribution_plot(include_all_features=True)
        """
        explainer = self._require_explainer("contribution_plot")

        if not include_all_features:
            if feature is None:
                raise ValueError("contribution_plot block requires 'feature' when include_all_features=False.")
            fig = explainer.plot.contribution_plot(feature, label=label, max_points=max_points or self.max_points)
            for trace in fig.data:
                if trace.type == "bar":
                    trace.marker.color = "lightgrey"
            return self._wrap_section_content(title or self._feature_label(feature), [self._plotly_pane(fig)])

        if getattr(explainer, "x_init", None) is None:
            raise ValueError("contribution_plot block with include_all_features=True requires explainer.x_init.")

        feature_names = list(explainer.x_init.columns)
        if not feature_names:
            return self._wrap_section_content(title, [pn.pane.Markdown("No feature available.")])

        sorted_features = sorted(
            feature_names,
            key=lambda current_feature: (str(self._feature_label(current_feature)).lower(), str(current_feature)),
        )

        feature_panels: dict[str, pn.viewable.Viewable] = {}
        for feature_name in sorted_features:
            fig = explainer.plot.contribution_plot(feature_name, label=label, max_points=max_points or self.max_points)
            for trace in fig.data:
                if trace.type == "bar":
                    trace.marker.color = "lightgrey"

            base_label = str(self._feature_label(feature_name))
            label_text = base_label
            suffix = 2
            while label_text in feature_panels:
                label_text = f"{base_label} ({suffix})"
                suffix += 1
            feature_panels[label_text] = self._plotly_pane(fig)

        feature_select = pn.widgets.Select(
            name="Feature",
            options=list(feature_panels.keys()),
            value=next(iter(feature_panels)),
            sizing_mode="stretch_width",
        )
        selected_panel = pn.bind(lambda selected: feature_panels[selected], feature_select)

        resolved_title = title or "Features contribution plots"
        return self._wrap_section_content(resolved_title, [feature_select, selected_panel])

    def block_interactions_plot(
        self,
        title: str = "",
        col1: str | None = None,
        col2: str | None = None,
        max_points: int | None = None,
    ) -> pn.Column:
        """Render an interactions plot between two features.

        Parameters
        ----------
        title : str, default=""
            Optional section title.
        col1 : str or None, default=None
            First feature. If None, the method picks a default interaction pair.
        col2 : str or None, default=None
            Second feature. If None, the method picks a default interaction pair.
        max_points : int or None, default=None
            Maximum number of points used by the plotting backend.

        Returns
        -------
        pn.Column
            Panel column containing the interactions plot.

        Examples
        --------
        >>> runtime.block_interactions_plot(col1="age", col2="income")
        """
        explainer = self._require_explainer("interactions_plot")
        feature_one, feature_two = self._resolve_interaction_pair(col1, col2)
        fig = explainer.plot.interactions_plot(
            col1=feature_one, col2=feature_two, max_points=max_points or self.max_points
        )
        resolved_title = title or f"{self._feature_label(feature_one)} / {self._feature_label(feature_two)}"
        return self._wrap_section_content(resolved_title, [self._plotly_pane(fig)])

    def block_target_distribution(
        self,
        title: str = "",
        width: int = 700,
        height: int = 500,
    ) -> pn.Column:
        """Render prediction-versus-true target distribution.

        Parameters
        ----------
        title : str, default=""
            Optional section title.
        width : int, default=700
            Plot width in pixels.
        height : int, default=500
            Plot height in pixels.

        Returns
        -------
        pn.Column
            Panel column containing the target distribution comparison plot.

        Examples
        --------
        >>> runtime.block_target_distribution()
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
        return self._wrap_section_content(title or "Target distribution", [self._plotly_pane(fig)])

    def block_target_analysis(
        self,
        title: str = "Target analysis",
        show_train: bool = True,
        width: int = 700,
        height: int = 500,
    ) -> pn.Column:
        """Render target statistics and target distribution analysis.

        Parameters
        ----------
        title : str, default="Target analysis"
            Section title displayed above target analysis elements.
        show_train : bool, default=True
            Whether training target information is included.
        width : int, default=700
            Plot width in pixels.
        height : int, default=500
            Plot height in pixels.

        Returns
        -------
        pn.Column
            Panel column combining a target statistics table and distribution plot.

        Examples
        --------
        >>> runtime.block_target_analysis(show_train=False)
        """
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

        distribution_frames = [pd.DataFrame({target_name: y_test_series}).assign(data_train_test="test")]
        if y_train_series is not None:
            distribution_frames.append(pd.DataFrame({target_name: y_train_series}).assign(data_train_test="train"))
        distribution_df = pd.concat(distribution_frames, ignore_index=True)

        fig = plot_distribution(
            df_all=distribution_df,
            col=target_name,
            hue="data_train_test",
            colors_dict=self._feature_distribution_colors(),
            width=width,
            height=height,
        )

        dtype_label = str(series_dtype(y_test_series))
        content = [
            pn.pane.Markdown(f"**{target_name}** ({dtype_label})"),
            pn.Row(
                pn.pane.DataFrame(
                    target_stats,
                    width_policy="min",
                    css_classes=["kv-table", "fit-content-table"],
                ),
                self._plotly_pane(fig),
                sizing_mode="stretch_width",
            ),
        ]
        return self._wrap_section_content(title, content)

    def block_confusion_matrix(self, title: str = "") -> pn.Column:
        """Render confusion matrix for classification predictions.

        Parameters
        ----------
        title : str, default=""
            Optional section title.

        Returns
        -------
        pn.Column
            Panel column containing the confusion matrix plot.

        Examples
        --------
        >>> runtime.block_confusion_matrix()
        """
        explainer = self._require_explainer("confusion_matrix")
        if self.y_test is None or self.y_pred is None:
            raise ValueError("confusion_matrix block requires y_test and predicted values from the explainer.")
        fig = plot_confusion_matrix(y_true=self.y_test, y_pred=self.y_pred, colors_dict=explainer.colors_dict)
        return self._wrap_section_content(title or "Confusion matrix", [self._plotly_pane(fig)])

    def block_univariate_analysis(
        self,
        title: str = "Univariate analysis",
        show_train: bool = True,
    ) -> pn.Column:
        """Render per-feature univariate analysis with interactive selection.

        Parameters
        ----------
        title : str, default="Univariate analysis"
            Section title displayed above selector and analysis panel.
        show_train : bool, default=True
            Whether train statistics are shown alongside prediction statistics.

        Returns
        -------
        pn.Column
            Panel column with a feature selector and corresponding stats/plot content.

        Examples
        --------
        >>> runtime.block_univariate_analysis()
        """
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
        feature_panels: dict[str, pn.viewable.Viewable] = {}

        for col_label in list_cols_labels:
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
            dtype_label = str(series_dtype(df[col]))
            tab_body = pn.Column(
                pn.pane.Markdown(f"**{col_label}** ({dtype_label})"),
                pn.Row(
                    pn.pane.DataFrame(
                        col_stats,
                        width_policy="min",
                        css_classes=["kv-table", "fit-content-table"],
                    ),
                    self._plotly_pane(fig),
                    sizing_mode="stretch_width",
                ),
                sizing_mode="stretch_width",
            )

            base_label = str(col_label)
            label_text = base_label
            suffix = 2
            while label_text in feature_panels:
                label_text = f"{base_label} ({suffix})"
                suffix += 1
            feature_panels[label_text] = tab_body

        if len(feature_panels) == 0:
            return self._wrap_section_content(title, [pn.pane.Markdown("No feature available.")])

        feature_select = pn.widgets.Select(
            name="Feature",
            options=list(feature_panels.keys()),
            value=next(iter(feature_panels)),
            sizing_mode="stretch_width",
        )
        selected_panel = pn.bind(lambda selected: feature_panels[selected], feature_select)

        return self._wrap_section_content(title, [feature_select, selected_panel])

    def _preprocess_train_data(self, x_train: pd.DataFrame | None) -> pd.DataFrame | None:
        if x_train is None or self.explainer is None:
            return x_train
        x_train_pre = inverse_transform(x_train, self.explainer.preprocessing)
        x_train_pre = handle_categorical_missing(x_train_pre)
        if self.explainer.postprocessing:
            x_train_pre = apply_postprocessing(x_train_pre, self.explainer.postprocessing)
        return x_train_pre

    @staticmethod
    def _get_values_and_name(y: pd.DataFrame | pd.Series | list | None, default_name: str) -> tuple[object, str | None]:
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
        if self.explainer is None:
            raise ValueError(f"{block_type} block requires an explainer on the report instance.")
        return self.explainer

    def _require_train_test_data(self, block_type: str) -> None:
        if self.df_train_test is None:
            raise ValueError(f"{block_type} block requires x_train and explainer.x_init data on the report instance.")

    def _resolve_interaction_pair(self, col1: str | None, col2: str | None) -> tuple[str, str]:
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
        if self.explainer is None:
            return feature
        return self.explainer.features_dict.get(feature, feature)

    def _feature_distribution_colors(self) -> dict:
        explainer = self._require_explainer("feature_distribution")
        return explainer.colors_dict["report_feature_distribution"]

    @staticmethod
    def _performance_distribution_colors() -> dict:
        return {"pred": "#2255aa", "true": "#f4c000"}

    @staticmethod
    def _plotly_pane(fig) -> pn.pane.Plotly:
        return make_plotly_pane(fig)

    @staticmethod
    def _coerce_viewable(item: Any) -> pn.viewable.Viewable:
        if isinstance(item, pn.viewable.Viewable):
            return item
        if isinstance(item, str):
            return pn.pane.Markdown(item)
        return pn.panel(item)

    def _wrap_section_content(self, title: str, body_items: Any) -> pn.Column:
        items = body_items if isinstance(body_items, list) else [body_items]
        blocks: list[pn.viewable.Viewable] = []
        if title:
            heading_prefix = "###" if getattr(self, "_inside_group", False) else "#"
            blocks.append(pn.pane.Markdown(f"{heading_prefix} {title}", css_classes=["section-title"]))
        blocks.extend(self._coerce_viewable(item) for item in items if item is not None)
        return pn.Column(*blocks, css_classes=["section-block"], sizing_mode="stretch_width")
