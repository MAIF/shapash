"""Block implementations and report data helpers for smart reports."""

from __future__ import annotations

import pandas as pd

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
        sub = f'<div class="shapash-callout"><p>{subtitle}</p></div>' if subtitle else ""
        return f'<div class="main-header"><h1>{title}</h1>{sub}</div>'

    def block_text(self, title: str = "", body: str = "", color: str = "gray") -> str:
        h2 = f'<h2 class="section-title">{title}</h2>' if title else ""
        return f'<div class="content-block">{h2}<p>{body}</p></div>'

    def block_key_value(self, title: str = "", items: dict | None = None, color: str = "gold") -> str:
        items = items or {}
        rows = self._render_key_value_rows(items)
        h2 = f'<h2 class="section-title">{title}</h2>' if title else ""
        return f'<div class="content-block">{h2}<table class="kv-table">{rows}</table></div>'

    def block_badge_row(self, title: str = "", badges: list | None = None) -> str:
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
        return f'<div class="shapash-callout"><p>{body}</p></div>'

    def block_divider(self, label: str = "") -> str:
        return '<div class="shapash-divider"></div>'

    def block_global_analysis(self, title: str = "", color: str = "gray") -> str:
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
