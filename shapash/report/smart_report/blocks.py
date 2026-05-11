"""Block implementations and report data helpers for smart reports."""

from __future__ import annotations

import importlib
import importlib.metadata
from pathlib import Path

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
    "gold": {"bg": "#ffffff", "border": "#ffbb00", "title": "#ccac00", "text": "#333333"},
    "blue": {"bg": "#ffffff", "border": "#2255aa", "title": "#2255aa", "text": "#333333"},
    "gray": {"bg": "#ffffff", "border": "#eeeeee", "title": "#666666", "text": "#666666"},
    "orange": {"bg": "#fff9e6", "border": "#ffbb00", "title": "#cc8833", "text": "#444444"},
}


class ReportBlockMixin:
    """Reusable block rendering and data preparation helpers."""

    def block_header(self, title: str = "Report", subtitle: str = "") -> str:
        """Return the HTML for the report header and its optional subtitle callout."""
        sub = f'<div class="shapash-callout"><p>{subtitle}</p></div>' if subtitle else ""
        return f'<div class="main-header"><h1>{title}</h1>{sub}</div>'

    def block_text(self, title: str = "", body: str = "", color: str = "gray") -> str:
        """Return the HTML for a text section with an optional title."""
        h2 = f'<h2 class="section-title">{title}</h2>' if title else ""
        return f'<div class="content-block">{h2}<p>{body}</p></div>'

    def block_project_information(
        self,
        title: str = "Project information",
        color: str = "gray",
        project_info_file: str = "",
    ) -> str:
        """Return project information loaded from an external YAML file."""
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

        sections_html = []
        for section_name, section_values in project_info.items():
            if not isinstance(section_values, dict):
                continue
            rows = self._render_key_value_rows(section_values)
            sections_html.append(
                f'<div class="content-block"><h3 class="section-title" style="font-size:1.2rem">{section_name}</h3>'
                f'<table class="kv-table">{rows}</table></div>'
            )

        return self._wrap_section_content(title, "".join(sections_html))

    def block_key_value(self, title: str = "", items: dict | None = None, color: str = "gold") -> str:
        """Return the HTML for a table of key-value pairs."""
        items = items or {}
        rows = self._render_key_value_rows(items)
        h2 = f'<h2 class="section-title">{title}</h2>' if title else ""
        return f'<div class="content-block">{h2}<table class="kv-table">{rows}</table></div>'

    def block_badge_row(self, title: str = "", badges: list | None = None) -> str:
        """Return the HTML for a row of badge-style metrics."""
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
        """Return the HTML for a highlighted callout paragraph."""
        return f'<div class="shapash-callout"><p>{body}</p></div>'

    def block_divider(self, label: str = "") -> str:
        """Return the HTML for a visual divider between report sections."""
        return '<div class="shapash-divider"></div>'

    def block_global_analysis(self, title: str = "", color: str = "gray") -> str:
        """Return the HTML for the global dataset statistics comparison table.

        Requires prediction data on the report instance and includes training
        data statistics when training data is available.
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
        """Return model metadata and parameters in a notebook-parity layout."""
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
        """Return a feature/target relationship plot on training data."""
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
        """Return training-only correlation heatmap (legacy notebook block name)."""
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
        """Return a badge row with configured evaluation metrics computed on y_test/y_pred."""
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
        """Return a scatter plot of predictions versus true target values."""
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
        """Return the HTML for a feature distribution plot across dataset splits.

        The feature must be present in the prepared train/test dataframe stored
        on the report instance.
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
        """Return the HTML for the explainer correlation plot.

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
        """Return the HTML for the explainer feature-importance plot."""
        explainer = self._require_explainer("feature_importance")
        fig = explainer.plot.features_importance(label=label)
        return self._wrap_section_content(title, self._plotly_html(fig))

    def block_contribution_plot(
        self,
        feature: str,
        title: str = "",
        color: str = "green",
        label=None,
        max_points: int | None = None,
    ) -> str:
        """Return the HTML for a feature contribution plot.

        Requires an explainer with contribution values and uses the configured
        maximum point count when no explicit limit is provided.
        """
        explainer = self._require_explainer("contribution_plot")
        fig = explainer.plot.contribution_plot(feature, label=label, max_points=max_points or self.max_points)
        for trace in fig.data:
            if trace.type == "bar":
                trace.marker.color = "lightgrey"
        return self._wrap_section_content(title or self._feature_label(feature), self._plotly_html(fig))

    def block_interactions_plot(
        self,
        title: str = "",
        color: str = "green",
        col1: str | None = None,
        col2: str | None = None,
        max_points: int | None = None,
    ) -> str:
        """Return the HTML for an interaction plot between two features.

        If no feature pair is provided, the strongest available interaction is
        selected from the explainer output.
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
        """Return the HTML for the true-versus-predicted target distribution plot.

        Requires both ground-truth targets and predicted values on the report
        instance.
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

    def block_confusion_matrix(self, title: str = "", color: str = "orange") -> str:
        """Return the HTML for a classification confusion matrix.

        Requires both ground-truth labels and predicted labels on the report
        instance.
        """
        explainer = self._require_explainer("confusion_matrix")
        if self.y_test is None or self.y_pred is None:
            raise ValueError("confusion_matrix block requires y_test and predicted values from the explainer.")
        fig = plot_confusion_matrix(y_true=self.y_test, y_pred=self.y_pred, colors_dict=explainer.colors_dict)
        return self._wrap_section_content(title or "Confusion matrix", self._plotly_html(fig))

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
            raise ValueError('"data_train_test" column must be renamed as it is used in ReportBase')
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
        return {"pred": "#2255aa", "true": "#ffbb00"}

    @staticmethod
    def _plotly_html(fig) -> str:
        return render_plotly_pane_html(fig)

    @staticmethod
    def _render_key_value_rows(items: dict) -> str:
        return "".join(
            f'<tr><td class="kv-key"><span class="kv-key-label">{key}</span><span class="kv-key-sep"> :</span></td>'
            f'<td class="kv-val">{value}</td></tr>'
            for key, value in items.items()
        )

    @staticmethod
    def _wrap_section_content(title: str, body_html: str) -> str:
        parts = []
        if title:
            parts.append(f'<h2 class="section-title">{title}</h2>')
        parts.append(body_html)
        return f'<div class="section-block">{"".join(parts)}</div>'
