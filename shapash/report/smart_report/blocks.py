"""Block implementations and report data helpers for smart reports."""

from __future__ import annotations

import numpy as np
import pandas as pd
from bokeh.models import BasicTicker, ColorBar, ColumnDataSource, HoverTool, LinearColorMapper, TabPanel, Tabs
from bokeh.palettes import RdYlBu11, YlOrRd9
from bokeh.plotting import figure
from bokeh.transform import dodge, jitter
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import confusion_matrix

from shapash.report.data_analysis import perform_global_dataframe_analysis
from shapash.report.smart_report.panel_support import render_bokeh_pane_html
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

        fig = self._feature_distribution_bokeh(
            feature=feature,
            dataset_split=dataset_split,
            prediction_label=prediction_label,
            training_label=training_label,
            width=width,
            height=height,
        )
        return self._wrap_section_content(title or self._feature_label(feature), self._bokeh_html(fig))

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
        resolved_width = width or (900 if len(self.df_train_test["data_train_test"].unique()) > 1 else 500)
        fig = self._correlations_bokeh(
            df=self.df_train_test,
            split_col="data_train_test",
            max_features=max_features,
            width=resolved_width,
            height=height,
            title=title,
        )
        return self._wrap_section_content("", self._bokeh_html(fig))

    def block_feature_importance(self, title: str = "", color: str = "green", label=None) -> str:
        """Return the HTML for the explainer feature-importance plot."""
        explainer = self._require_explainer("feature_importance")
        fig = self._feature_importance_bokeh(explainer=explainer, title=title)
        return self._wrap_section_content("", self._bokeh_html(fig))

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
        fig = self._contribution_bokeh(
            explainer=explainer,
            feature=feature,
            max_points=max_points or self.max_points,
            title=title or self._feature_label(feature),
        )
        return self._wrap_section_content("", self._bokeh_html(fig))

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
        fig = self._interactions_bokeh(
            explainer=explainer,
            col1=feature_one,
            col2=feature_two,
            max_points=max_points or self.max_points,
            title=title or f"{self._feature_label(feature_one)} / {self._feature_label(feature_two)}",
        )
        return self._wrap_section_content("", self._bokeh_html(fig))

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
        fig = self._target_distribution_bokeh(
            df_target=df_target,
            target_name=target_name,
            width=width,
            height=height,
            title=title or "Target distribution",
        )
        return self._wrap_section_content("", self._bokeh_html(fig))

    def block_confusion_matrix(self, title: str = "", color: str = "orange") -> str:
        """Return the HTML for a classification confusion matrix.

        Requires both ground-truth labels and predicted labels on the report
        instance.
        """
        explainer = self._require_explainer("confusion_matrix")
        if self.y_test is None or self.y_pred is None:
            raise ValueError("confusion_matrix block requires y_test and predicted values from the explainer.")
        fig = self._confusion_matrix_bokeh(
            y_true=self.y_test,
            y_pred=self.y_pred,
            title=title or "Confusion matrix",
        )
        return self._wrap_section_content("", self._bokeh_html(fig))

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
    def _palette_color(index: int, fallback: list[str]) -> str:
        return fallback[index % len(fallback)]

    def _resolve_split_colors(self, split_values: list[str], colors_dict: dict | None = None) -> dict[str, str]:
        base = colors_dict or {}
        fallback = ["#2255aa", "#ffbb00", "#666666", "#44aa99", "#cc6677"]
        return {
            split: base.get(split, self._palette_color(idx, fallback))
            for idx, split in enumerate(split_values)
        }

    def _feature_distribution_bokeh(
        self,
        feature: str,
        dataset_split: str,
        prediction_label: str,
        training_label: str,
        width: int,
        height: int,
    ):
        df_plot = self.df_train_test[[feature, dataset_split]].dropna().copy()
        if df_plot.empty:
            raise ValueError(f"No data available for feature '{feature}' after dropping missing values.")

        label_map = {"test": prediction_label, "train": training_label}
        split_values = [str(val) for val in df_plot[dataset_split].dropna().unique().tolist()]
        split_colors = self._resolve_split_colors(split_values, self._feature_distribution_colors())

        tools = "pan,wheel_zoom,box_zoom,reset,save"
        title = self._feature_label(feature)
        if is_numeric_dtype(df_plot[feature]):
            bins = min(max(10, int(np.sqrt(len(df_plot)))), 40)
            p = figure(width=width, height=height, title=title, tools=tools)
            p.yaxis.axis_label = "Count"
            p.xaxis.axis_label = title

            for split in split_values:
                subset = df_plot[df_plot[dataset_split].astype(str) == split][feature].astype(float)
                if subset.empty:
                    continue
                hist, edges = np.histogram(subset, bins=bins)
                source = ColumnDataSource(
                    data={
                        "left": edges[:-1],
                        "right": edges[1:],
                        "top": hist,
                        "split": [label_map.get(split, split)] * len(hist),
                    }
                )
                renderer = p.quad(
                    source=source,
                    top="top",
                    bottom=0,
                    left="left",
                    right="right",
                    fill_color=split_colors[split],
                    line_color=split_colors[split],
                    fill_alpha=0.35,
                    legend_label=label_map.get(split, split),
                    muted_alpha=0.1,
                )
                p.add_tools(
                    HoverTool(
                        renderers=[renderer],
                        tooltips=[
                            ("Split", "@split"),
                            ("Bin", "@left{0.000} - @right{0.000}"),
                            ("Count", "@top"),
                        ],
                    )
                )

            p.legend.click_policy = "mute"
            return p

        categories = (
            df_plot[feature]
            .astype(str)
            .value_counts()
            .head(15)
            .index.tolist()
        )
        filtered = df_plot[df_plot[feature].astype(str).isin(categories)].copy()
        p = figure(x_range=categories, width=width, height=height, title=title, tools=tools)
        p.yaxis.axis_label = "Count"
        p.xaxis.axis_label = title

        n_splits = max(len(split_values), 1)
        bar_width = 0.8 / n_splits
        start = -0.4 + (bar_width / 2)

        for idx, split in enumerate(split_values):
            subset = filtered[filtered[dataset_split].astype(str) == split]
            counts = subset[feature].astype(str).value_counts().reindex(categories, fill_value=0)
            source = ColumnDataSource(
                data={
                    "category": categories,
                    "count": counts.values,
                    "split": [label_map.get(split, split)] * len(categories),
                }
            )
            renderer = p.vbar(
                x=dodge("category", start + idx * bar_width, range=p.x_range),
                top="count",
                width=bar_width,
                source=source,
                color=split_colors[split],
                line_color=split_colors[split],
                fill_alpha=0.8,
                legend_label=label_map.get(split, split),
                muted_alpha=0.1,
            )
            p.add_tools(
                HoverTool(
                    renderers=[renderer],
                    tooltips=[
                        ("Split", "@split"),
                        ("Category", "@category"),
                        ("Count", "@count"),
                    ],
                )
            )

        p.xaxis.major_label_orientation = 0.8
        p.legend.click_policy = "mute"
        return p

    def _correlation_heatmap_figure(self, corr: pd.DataFrame, title: str, width: int, height: int):
        corr = corr.fillna(0.0)
        labels = list(corr.columns)
        corr_long = corr.stack().rename("corr").reset_index().rename(columns={"level_0": "y", "level_1": "x"})
        source = ColumnDataSource(corr_long)
        mapper = LinearColorMapper(palette=list(reversed(RdYlBu11)), low=-1, high=1)

        p = figure(
            title=title,
            x_range=labels,
            y_range=list(reversed(labels)),
            width=width,
            height=height,
            tools="pan,wheel_zoom,box_zoom,reset,save",
        )
        renderer = p.rect(
            x="x",
            y="y",
            width=1,
            height=1,
            source=source,
            line_color=None,
            fill_color={"field": "corr", "transform": mapper},
        )
        p.add_tools(
            HoverTool(
                renderers=[renderer],
                tooltips=[("Feature X", "@x"), ("Feature Y", "@y"), ("Correlation", "@corr{0.000}")],
            )
        )
        p.xaxis.major_label_orientation = 0.9
        p.grid.grid_line_color = None
        p.add_layout(
            ColorBar(color_mapper=mapper, ticker=BasicTicker(desired_num_ticks=7), label_standoff=8, location=(0, 0)),
            "right",
        )
        return p

    def _correlations_bokeh(
        self,
        df: pd.DataFrame,
        split_col: str,
        max_features: int,
        width: int,
        height: int,
        title: str,
    ):
        numeric_cols = [col for col in df.select_dtypes(include="number").columns if col != split_col]
        if not numeric_cols:
            raise ValueError("No numeric feature available to compute correlations.")

        if max_features > 0:
            numeric_cols = numeric_cols[:max_features]

        split_values = [str(v) for v in df[split_col].dropna().unique().tolist()] if split_col in df.columns else []
        if not split_values:
            corr = df[numeric_cols].corr(numeric_only=True)
            return self._correlation_heatmap_figure(corr, title=title or "Correlations", width=width, height=height)

        panels = []
        for split in split_values:
            subset = df[df[split_col].astype(str) == split][numeric_cols]
            corr = subset.corr(numeric_only=True)
            panel_title = f"{title} - {split}" if title else f"Correlations - {split}"
            panels.append(TabPanel(child=self._correlation_heatmap_figure(corr, panel_title, width, height), title=split))
        return Tabs(tabs=panels)

    def _feature_importance_bokeh(self, explainer, title: str):
        if getattr(explainer, "features_imp", None) is None:
            explainer.compute_features_import()
        features_imp = explainer.features_imp
        if isinstance(features_imp, list):
            if not features_imp:
                raise ValueError("features_imp is empty.")
            features_imp = features_imp[0]

        ordered = features_imp.sort_values(ascending=False).head(20)
        labels = [explainer.features_dict.get(name, name) for name in ordered.index.tolist()]
        source = ColumnDataSource(data={"feature": list(reversed(labels)), "importance": list(reversed(ordered.values))})
        p = figure(
            title=title or "Feature importance",
            y_range=list(reversed(labels)),
            width=900,
            height=560,
            tools="pan,wheel_zoom,box_zoom,reset,save",
        )
        renderer = p.hbar(y="feature", right="importance", height=0.7, source=source, color="#ffbb00")
        p.xaxis.axis_label = "Importance"
        p.yaxis.axis_label = "Feature"
        p.grid.grid_line_alpha = 0.25
        p.add_tools(HoverTool(renderers=[renderer], tooltips=[("Feature", "@feature"), ("Importance", "@importance{0.00}")]))
        return p

    @staticmethod
    def _select_contrib_frame(contributions, label=None):
        if isinstance(contributions, list):
            if not contributions:
                raise ValueError("Contributions list is empty.")
            if isinstance(label, int) and 0 <= label < len(contributions):
                return contributions[label]
            return contributions[0]
        return contributions

    def _contribution_bokeh(self, explainer, feature: str, max_points: int, title: str):
        contrib_df = self._select_contrib_frame(explainer.contributions)
        if feature not in contrib_df.columns or feature not in explainer.x_init.columns:
            raise ValueError(f"Unknown feature '{feature}' for contribution_plot block.")

        plot_df = pd.DataFrame({
            "feature_value": explainer.x_init[feature],
            "contribution": contrib_df[feature],
        }).dropna()
        if max_points and len(plot_df) > max_points:
            plot_df = plot_df.sample(n=max_points, random_state=0)

        tools = "pan,wheel_zoom,box_zoom,reset,save"
        if is_numeric_dtype(plot_df["feature_value"]):
            source = ColumnDataSource(plot_df)
            p = figure(title=title, width=900, height=500, tools=tools)
            renderer = p.scatter("feature_value", "contribution", source=source, size=6, alpha=0.55, color="#777777")
            p.xaxis.axis_label = self._feature_label(feature)
            p.yaxis.axis_label = "Contribution"
            p.add_tools(
                HoverTool(
                    renderers=[renderer],
                    tooltips=[(self._feature_label(feature), "@feature_value"), ("Contribution", "@contribution{0.0000}")],
                )
            )
            return p

        plot_df["feature_value"] = plot_df["feature_value"].astype(str)
        top = plot_df["feature_value"].value_counts().head(20).index.tolist()
        plot_df = plot_df[plot_df["feature_value"].isin(top)]
        source = ColumnDataSource(plot_df)
        p = figure(title=title, x_range=top, width=900, height=500, tools=tools)
        renderer = p.scatter(
            x=jitter("feature_value", width=0.35, range=p.x_range),
            y="contribution",
            source=source,
            size=6,
            alpha=0.5,
            color="#777777",
        )
        p.xaxis.major_label_orientation = 0.8
        p.xaxis.axis_label = self._feature_label(feature)
        p.yaxis.axis_label = "Contribution"
        p.add_tools(
            HoverTool(
                renderers=[renderer],
                tooltips=[(self._feature_label(feature), "@feature_value"), ("Contribution", "@contribution{0.0000}")],
            )
        )
        return p

    def _interactions_bokeh(self, explainer, col1: str, col2: str, max_points: int, title: str):
        if col1 not in explainer.x_init.columns or col2 not in explainer.x_init.columns:
            raise ValueError(f"Unknown interaction pair '{col1}', '{col2}'.")

        plot_df = explainer.x_init[[col1, col2]].dropna().copy()
        if max_points and len(plot_df) > max_points:
            plot_df = plot_df.sample(n=max_points, random_state=0)

        source = ColumnDataSource(plot_df.rename(columns={col1: "x", col2: "y"}))
        p = figure(title=title, width=900, height=500, tools="pan,wheel_zoom,box_zoom,reset,save")
        renderer = p.scatter("x", "y", source=source, size=6, alpha=0.55, color="#2255aa")
        p.xaxis.axis_label = self._feature_label(col1)
        p.yaxis.axis_label = self._feature_label(col2)
        p.add_tools(HoverTool(renderers=[renderer], tooltips=[(self._feature_label(col1), "@x"), (self._feature_label(col2), "@y")]))
        return p

    def _target_distribution_bokeh(self, df_target: pd.DataFrame, target_name: str, width: int, height: int, title: str):
        split_col = "_dataset"
        split_values = [str(v) for v in df_target[split_col].dropna().unique().tolist()]
        label_map = {"pred": "pred", "true": "true"}
        split_colors = self._resolve_split_colors(split_values, self._performance_distribution_colors())
        tools = "pan,wheel_zoom,box_zoom,reset,save"

        if is_numeric_dtype(df_target[target_name]):
            bins = min(max(10, int(np.sqrt(len(df_target)))), 40)
            p = figure(width=width, height=height, title=title, tools=tools)
            p.yaxis.axis_label = "Count"
            p.xaxis.axis_label = target_name
            for split in split_values:
                subset = df_target[df_target[split_col].astype(str) == split][target_name].astype(float)
                if subset.empty:
                    continue
                hist, edges = np.histogram(subset, bins=bins)
                source = ColumnDataSource(
                    data={
                        "left": edges[:-1],
                        "right": edges[1:],
                        "top": hist,
                        "split": [label_map.get(split, split)] * len(hist),
                    }
                )
                renderer = p.quad(
                    source=source,
                    top="top",
                    bottom=0,
                    left="left",
                    right="right",
                    fill_color=split_colors[split],
                    line_color=split_colors[split],
                    fill_alpha=0.35,
                    legend_label=label_map.get(split, split),
                    muted_alpha=0.1,
                )
                p.add_tools(
                    HoverTool(
                        renderers=[renderer],
                        tooltips=[("Split", "@split"), ("Bin", "@left{0.000} - @right{0.000}"), ("Count", "@top")],
                    )
                )
            p.legend.click_policy = "mute"
            return p

        categories = df_target[target_name].astype(str).value_counts().head(15).index.tolist()
        filtered = df_target[df_target[target_name].astype(str).isin(categories)].copy()
        p = figure(x_range=categories, width=width, height=height, title=title, tools=tools)
        p.yaxis.axis_label = "Count"
        p.xaxis.axis_label = target_name

        n_splits = max(len(split_values), 1)
        bar_width = 0.8 / n_splits
        start = -0.4 + (bar_width / 2)
        for idx, split in enumerate(split_values):
            subset = filtered[filtered[split_col].astype(str) == split]
            counts = subset[target_name].astype(str).value_counts().reindex(categories, fill_value=0)
            source = ColumnDataSource(
                data={
                    "category": categories,
                    "count": counts.values,
                    "split": [label_map.get(split, split)] * len(categories),
                }
            )
            renderer = p.vbar(
                x=dodge("category", start + idx * bar_width, range=p.x_range),
                top="count",
                width=bar_width,
                source=source,
                color=split_colors[split],
                line_color=split_colors[split],
                fill_alpha=0.8,
                legend_label=label_map.get(split, split),
                muted_alpha=0.1,
            )
            p.add_tools(
                HoverTool(
                    renderers=[renderer],
                    tooltips=[("Split", "@split"), ("Category", "@category"), ("Count", "@count")],
                )
            )
        p.xaxis.major_label_orientation = 0.8
        p.legend.click_policy = "mute"
        return p

    def _confusion_matrix_bokeh(self, y_true, y_pred, title: str):
        true_series = pd.Series(y_true).astype(str)
        pred_series = pd.Series(y_pred).astype(str)
        classes = sorted(set(true_series.unique()) | set(pred_series.unique()))
        cm = confusion_matrix(true_series, pred_series, labels=classes)
        cm_df = (
            pd.DataFrame(cm, index=classes, columns=classes)
            .stack()
            .rename("count")
            .reset_index()
            .rename(columns={"level_0": "true", "level_1": "pred"})
        )
        source = ColumnDataSource(cm_df)

        mapper = LinearColorMapper(palette=YlOrRd9, low=float(cm.min()), high=float(max(cm.max(), 1)))
        p = figure(
            title=title,
            x_range=classes,
            y_range=list(reversed(classes)),
            width=750,
            height=650,
            tools="pan,wheel_zoom,box_zoom,reset,save",
        )
        renderer = p.rect(
            x="pred",
            y="true",
            width=1,
            height=1,
            source=source,
            line_color="white",
            fill_color={"field": "count", "transform": mapper},
        )
        p.text(x="pred", y="true", text="count", source=source, text_align="center", text_baseline="middle")
        p.xaxis.axis_label = "Predicted"
        p.yaxis.axis_label = "True"
        p.xaxis.major_label_orientation = 0.8
        p.add_tools(HoverTool(renderers=[renderer], tooltips=[("True", "@true"), ("Predicted", "@pred"), ("Count", "@count")]))
        p.add_layout(
            ColorBar(color_mapper=mapper, ticker=BasicTicker(desired_num_ticks=6), label_standoff=8, location=(0, 0)),
            "right",
        )
        return p

    @staticmethod
    def _performance_distribution_colors() -> dict:
        return {"pred": "#2255aa", "true": "#ffbb00"}

    @staticmethod
    def _bokeh_html(fig) -> str:
        return render_bokeh_pane_html(fig)

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
