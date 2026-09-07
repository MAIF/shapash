"""Block implementations and report data helpers for smart reports."""

from __future__ import annotations

import importlib
import importlib.metadata
import inspect
import logging
from functools import wraps
from typing import Any, TypeAlias, cast

import numpy as np
import pandas as pd
import panel as pn

from shapash.plots.plot_evaluation_metrics import plot_confusion_matrix, plot_lift_curve
from shapash.plots.plot_univariate import plot_distribution
from shapash.report.common import compute_col_types, series_dtype
from shapash.report.core import _wrap_section_anchor
from shapash.report.data_analysis import perform_global_dataframe_analysis, perform_univariate_dataframe_analysis
from shapash.report.panel_support import _add_css_classes, _auto_style_viewable, _coerce_viewable
from shapash.report.validation import render_block_error, stats_to_table
from shapash.utils.transform import apply_postprocessing, handle_categorical_missing, inverse_transform

logger = logging.getLogger(__name__)

PALETTE = {
    "gold": {"bg": "#ffffff", "border": "#f4c000", "title": "#f4c000", "text": "#343736"},
    "blue": {"bg": "#ffffff", "border": "#2255aa", "title": "#2255aa", "text": "#343736"},
    "gray": {"bg": "#ffffff", "border": "#eeeeee", "title": "#666666", "text": "#666666"},
    "orange": {"bg": "#fff9e6", "border": "#f4c000", "title": "#cc8833", "text": "#444444"},
}

TARGET_DISTRIBUTION_COLORS = {"pred": "#2255aa", "true": "#f4c000"}
BlockContent: TypeAlias = tuple[str, list[Any]]
TargetValues: TypeAlias = np.ndarray[Any, Any] | list[Any]


def block(method):
    """Wrap block output in a standard report section container.

    Decorated methods can return either ``(title, body)`` or a bare body value.
    The body may be a single supported item or a list of supported items. Each
    item can be a string, a pandas ``DataFrame``, a Plotly figure, or a Panel
    viewable. Tuples inside the body are rendered as horizontal rows.
    """

    @wraps(method)
    def wrapped(self, *args, **kwargs):
        result = method(self, *args, **kwargs)

        # get block method results
        if isinstance(result, tuple) and len(result) == 2:
            title, body = result
        else:  # handle missing title
            body = result
            try:
                bound_args = inspect.signature(method).bind(self, *args, **kwargs)
                bound_args.apply_defaults()
                title_value = bound_args.arguments.get("title", "")
            except (TypeError, ValueError):
                title_value = kwargs.get("title", "")
            if isinstance(title_value, str) and title_value.strip():
                title = title_value.strip()

        items = body if isinstance(body, list) else [body]
        blocks: list[pn.viewable.Viewable] = []

        heading_prefix = "###" if getattr(self, "_inside_group", False) else "#"
        blocks.append(_add_css_classes(pn.pane.Markdown(f"{heading_prefix} {title}"), "section-title"))

        # Gestion de la grille row/columns du contenu à afficher
        for item in items:
            if isinstance(item, tuple):
                row = pn.Row(
                    *[
                        _auto_style_viewable(_coerce_viewable(i), method_name=method.__name__)
                        for i in item
                        if i is not None
                    ],
                    sizing_mode="stretch_width",
                )
            elif item is not None:
                row = _auto_style_viewable(_coerce_viewable(item), method_name=method.__name__)
            else:
                continue

            blocks.append(row)

        return pn.Column(*blocks, css_classes=["section-block"], sizing_mode="stretch_width")

    return wrapped


class ReportBlockMixin:
    """Base mixin providing built-in and user-extensible smart report blocks."""

    def __init__(
        self,
        explainer=None,
        x_train: pd.DataFrame | None = None,
        y_train: pd.Series | pd.DataFrame | list | None = None,
        y_test: pd.Series | pd.DataFrame | list | None = None,
        max_points: int = 200,
    ) -> None:
        self.explainer = explainer
        self.x_train_init = x_train
        self.x_train_pre = self._preprocess_train_data(x_train)
        self.x_init = getattr(explainer, "x_init", None)
        self.df_train_test = self._create_train_test_df(test=self.x_init, train=self.x_train_pre)
        self.y_train, self.target_name_train = self._get_values_and_name(y_train, "target")
        self.y_test, self.target_name_test = self._get_values_and_name(y_test, "target")
        self.target_name = self.target_name_train if self.target_name_train is not None else self.target_name_test
        self.max_points = max_points
        self._inside_group = False

        if explainer is not None:
            if explainer.y_pred is not None:
                self.y_pred, _ = self._get_values_and_name(explainer.y_pred, "prediction")
            else:
                self.y_pred = explainer.model.predict(explainer.x_encoded)
        else:
            self.y_pred = None

    def render_block(self, block_cfg: dict):
        """Dispatch one YAML block entry to the matching block_* method."""

        block_type = block_cfg.get("type", "")
        params = block_cfg.get("params", {})

        if block_type == "group":
            previous_inside_group = getattr(self, "_inside_group", False)
            self._inside_group = True
            try:
                children = [self.render_block(child_cfg) for child_cfg in block_cfg.get("blocks", [])]
            finally:
                self._inside_group = previous_inside_group
            children = [child for child in children if child is not None]
            group_title = params.get("title", "")
            section_id = block_cfg.get("_section_id")
            if group_title:
                group_content = pn.Column(
                    pn.pane.Markdown(f"## {group_title}", css_classes=["group-title"]),
                    *children,
                    sizing_mode="stretch_width",
                )
                return _wrap_section_anchor(group_content, section_id)
            return _wrap_section_anchor(pn.Column(*children, sizing_mode="stretch_width"), section_id)

        method = getattr(self, f"block_{block_type}", None)
        if method is None:
            if block_type == "custom":
                return self._render_custom(block_cfg)
            logger.warning("Unknown block type '%s' - skipped.", block_type)
            return None

        try:
            result = method(**params)
            if isinstance(result, pn.viewable.Viewable):
                return _wrap_section_anchor(result, block_cfg.get("_section_id"))
            raise TypeError(
                f"The return type of {method.__name__} is not a panel Viewable. Did you forget the @block decorator?"
            )
        except Exception as exc:
            logger.error("Block '%s' raised: %s", block_type, exc)
            return render_block_error(block_type, exc)

    def _render_custom(self, block_cfg: dict):
        """Call an arbitrary importable function."""
        func_path = block_cfg.get("function", "")
        params = block_cfg.get("params", {})
        try:
            mod_path, fn_name = func_path.rsplit(".", 1)
            fn = getattr(importlib.import_module(mod_path), fn_name)
            result = fn(self, **params)
            if isinstance(result, pn.viewable.Viewable):
                return result
            if isinstance(result, str):
                return pn.pane.Markdown(result)
            return pn.panel(result)
        except Exception as exc:
            logger.error("Custom block '%s' raised: %s", func_path, exc)
            return render_block_error(func_path, exc)

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

    @block
    def block_text(self, title: str = "", content: dict[str, str] | str | None = None) -> BlockContent:
        """Render a key/value text list from YAML items.

        Parameters
        ----------
        title : str, default=""
            Optional section title.
        content : dict[str, str], str or None, default=None
            Dict of entries. For example ``{"version": "0.7", "name": "My project"}``.
            Or markdown text.

        Returns
        -------
        tuple[str, list[str]]
            Section title and markdown content rendered by the @block decorator.
        """
        if not content:
            return title, ["No information available."]

        lines: list[str] = []
        if isinstance(content, dict):
            for key, value in content.items():
                lines.append(f"**{key}** : {value}")
        else:
            lines.append(str(content))

        # Use markdown hard line breaks so each key/value appears on its own line.
        return title, ["  \n".join(lines)]

    @block
    def block_badge_row(self, title: str = "", badges: list | None = None) -> BlockContent:
        """Render a row of summary badges.

        Parameters
        ----------
        title : str, default=""
            Optional section title.
        badges : list or None, default=None
            List of dictionaries with keys such as ``label``, ``value``, and ``color``.

        Returns
        -------
        tuple[str, list[pn.viewable.Viewable]]
            Section title and badge row content rendered by the @block decorator.

        Examples
        --------
        >>> runtime.block_badge_row(badges=[{"label": "AUC", "value": "0.89", "color": "blue"}])
        """
        if badges is None:
            badges = []
        pills: list[pn.viewable.Viewable] = []
        for badge in badges:
            color_name = badge.get("color", "gray")
            if color_name not in PALETTE:
                color_name = "gray"
            pills.append(
                pn.pane.Markdown(
                    f"**{badge.get('label', '')}**: {badge.get('value', '')}",
                    css_classes=[f"badge-pill-{color_name}"],
                )
            )

        return title, [tuple(pills)]

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

    @block
    def block_global_analysis(self, title: str = "") -> BlockContent:
        """Render global summary statistics for prediction and training datasets.

        Parameters
        ----------
        title : str, default=""
            Optional section title.

        Returns
        -------
        tuple[str, list[pn.viewable.Viewable]]
            Section title and statistics table content rendered by the @block decorator.

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
        return title, [stats_table]

    @block
    def block_model_analysis(self, title: str = "Model information") -> BlockContent:
        """Render model metadata and parameter tables.

        Parameters
        ----------
        title : str, default="Model information"
            Section title displayed above model details.

        Returns
        -------
        tuple[str, list[pn.viewable.Viewable]]
            Section title and model details content rendered by the @block decorator.

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

        def _truncate(value: Any, max_len: int) -> str:
            text = str(value)
            return text if len(text) <= max_len else text[: max_len - 3] + "..."

        if len(params_items) > 15:
            split_idx = len(params_items) // 2
            left_df = pd.DataFrame(
                {
                    "Parameter": [_truncate(key, 50) for key, _ in params_items[:split_idx]],
                    "Value": [_truncate(val, 300) for _, val in params_items[:split_idx]],
                }
            )
            right_df = pd.DataFrame(
                {
                    "Parameter": [_truncate(key, 50) for key, _ in params_items[split_idx:]],
                    "Value": [_truncate(val, 300) for _, val in params_items[split_idx:]],
                }
            )
            params_table = (left_df, pn.Spacer(width=24), right_df)
        else:
            params_df = pd.DataFrame(
                {
                    "Parameter": [_truncate(key, 50) for key, _ in params_items],
                    "Value": [_truncate(val, 300) for _, val in params_items],
                }
            )
            params_table = params_df

        content: list[Any] = [
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
            params_table,
        ]

        return title, content

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
        if metrics is None:
            metrics = []
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

    @block
    def block_feature_distribution(
        self,
        feature: str,
        title: str = "",
        dataset_split: str = "data_train_test",
        width: int = 700,
        height: int = 500,
    ) -> BlockContent:
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
        tuple[str, list[pn.viewable.Viewable]]
            Section title and feature distribution viewable rendered by the @block decorator.

        Examples
        --------
        >>> runtime.block_feature_distribution(feature="age")
        """
        df_train_test = self._require_train_test_data("feature_distribution")
        if feature not in df_train_test.columns:
            raise ValueError(f"Unknown feature '{feature}' for feature_distribution block.")

        fig = plot_distribution(
            df_all=df_train_test,
            col=feature,
            hue=dataset_split,
            colors_dict=self._feature_distribution_colors(),
            width=width,
            height=height,
        )
        if title is None:
            return self._feature_label(feature), [fig]
        return title, [fig]

    @block
    def block_correlations_plot(
        self,
        title: str = "",
        max_features: int = 20,
        width: int | None = None,
        height: int = 500,
    ) -> BlockContent:
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
        tuple[str, list[pn.viewable.Viewable]]
            Section title and correlations plot content rendered by the @block decorator.

        Examples
        --------
        >>> runtime.block_correlations_plot(max_features=15)
        """
        df_train_test = self._require_train_test_data("correlations_plot")
        explainer = self._require_explainer("correlations_plot")
        if width is None:
            if len(df_train_test["data_train_test"].unique()) > 1:
                resolved_width = 900
            else:
                resolved_width = 500
        else:
            resolved_width = width
        fig = explainer.plot.correlations_plot(
            df_train_test,
            optimized=True,
            facet_col="data_train_test",
            max_features=max_features,
            width=resolved_width,
            height=height,
        )
        return title, [fig]

    @block
    def block_feature_importance(self, title: str = "", label=None) -> BlockContent:
        """Render global feature importance.

        Parameters
        ----------
        title : str, default=""
            Optional section title.
        label : Any, default=None
            Optional class/target label for label-specific importance.

        Returns
        -------
        tuple[str, list[pn.viewable.Viewable]]
            Section title and feature-importance content rendered by the @block decorator.

        Examples
        --------
        >>> runtime.block_feature_importance()
        """
        explainer = self._require_explainer("feature_importance")
        fig = explainer.plot.features_importance(label=label)
        return title, [fig]

    @block
    def block_contribution_plot(
        self,
        feature: str | None = None,
        title: str = "",
        label=None,
        max_points: int | None = None,
        include_all_features: bool = False,
    ) -> BlockContent:
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
        tuple[str, list[pn.viewable.Viewable]]
            Section title and contribution content rendered by the @block decorator.

        Examples
        --------
        >>> runtime.block_contribution_plot(feature="age")
        >>> runtime.block_contribution_plot(include_all_features=True)
        """
        explainer = self._require_explainer("contribution_plot")

        if not include_all_features:
            if feature is None:
                raise ValueError("contribution_plot block requires 'feature' when include_all_features=False.")
            if max_points is None:
                effective_max_points = self.max_points
            else:
                effective_max_points = max_points
            fig = explainer.plot.contribution_plot(feature, label=label, max_points=effective_max_points)
            for trace in fig.data:
                if trace.type == "bar":
                    trace.marker.color = "lightgrey"
            if title is None:
                return self._feature_label(feature), [fig]
            return title, [fig]

        if getattr(explainer, "x_init", None) is None:
            raise ValueError("contribution_plot block with include_all_features=True requires explainer.x_init.")

        feature_names = list(explainer.x_init.columns)
        if not feature_names:
            return title, [pn.pane.Markdown("No feature available.")]

        sorted_features = sorted(
            feature_names,
            key=lambda current_feature: (str(self._feature_label(current_feature)).lower(), str(current_feature)),
        )

        feature_panels: dict[str, pn.viewable.Viewable] = {}
        for feature_name in sorted_features:
            if max_points is None:
                effective_max_points = self.max_points
            else:
                effective_max_points = max_points
            fig = explainer.plot.contribution_plot(feature_name, label=label, max_points=effective_max_points)
            for trace in fig.data:
                if trace.type == "bar":
                    trace.marker.color = "lightgrey"

            base_label = str(self._feature_label(feature_name))
            label_text = base_label
            suffix = 2
            while label_text in feature_panels:
                label_text = f"{base_label} ({suffix})"
                suffix += 1
            feature_panels[label_text] = fig

        feature_select = pn.widgets.Select(
            name="Feature",
            options=list(feature_panels.keys()),
            value=next(iter(feature_panels)),
            sizing_mode="stretch_width",
        )
        selected_panel = pn.panel(
            pn.bind(cast(Any, lambda selected: feature_panels[selected]), feature_select), sizing_mode="stretch_width"
        )

        if title is None:
            resolved_title = "Features contribution plots"
        else:
            resolved_title = title
        return resolved_title, [feature_select, selected_panel]

    @block
    def block_top_interactions_plot(
        self,
        title: str = "Top interactions plot",
        nb_top_interaction: int = 5,
        class_label: int | str = -1,
        max_points: int | None = None,
    ) -> BlockContent:
        """Render a plot for the top feature interaction pairs.

        Parameters
        ----------
        title : str, default="Top interactions plot"
            Section title displayed above the interaction figure.
        nb_top_interaction : int, default=5
            Number of top interactions to display.
        class_label : int or str, default=-1
            Optional class/target label used to compute and render class-specific interactions.
        max_points : int or None, default=None
            Maximum number of points used by the plotting backend.

        Returns
        -------
        tuple[str, list[pn.viewable.Viewable]]
            Section title and top-interactions plot content rendered by the @block decorator.

        Examples
        --------
        >>> runtime.block_top_interactions_plot(nb_top_interaction=3)
        """
        explainer = self._require_explainer("top_interactions_plot")
        if max_points is None:
            effective_max_points = self.max_points
        else:
            effective_max_points = max_points
        fig = explainer.plot.top_interactions_plot(
            nb_top_interactions=nb_top_interaction,
            label=class_label,
            max_points=effective_max_points,
        )
        return title, [fig]

    @block
    def block_target_distribution(
        self,
        title: str = "",
        width: int = 700,
        height: int = 500,
    ) -> BlockContent:
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
        tuple[str, list[pn.viewable.Viewable]]
            Section title and target distribution content rendered by the @block decorator.

        Examples
        --------
        >>> runtime.block_target_distribution()
        """
        self._require_explainer("target_distribution")
        if self.y_test is None or self.y_pred is None:
            raise ValueError("target_distribution block requires y_test and predicted values from the explainer.")

        if self.target_name is None:
            target_name = "target"
        else:
            target_name = self.target_name
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
            colors_dict=TARGET_DISTRIBUTION_COLORS,
            width=width,
            height=height,
        )
        if title is None:
            return "Target distribution", [fig]
        return title, [fig]

    @block
    def block_target_analysis(
        self,
        title: str = "Target analysis",
        show_train: bool = True,
        width: int = 700,
        height: int = 500,
    ) -> BlockContent:
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
        tuple[str, list[pn.viewable.Viewable]]
            Section title and target-analysis content rendered by the @block decorator.

        Examples
        --------
        >>> runtime.block_target_analysis(show_train=False)
        """
        if self.y_test is None:
            raise ValueError("target_analysis block requires y_test.")

        if self.target_name is None:
            target_name = "target"
        else:
            target_name = self.target_name
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
        fig.update_layout(
            title={
                **fig.layout.title.to_plotly_json(),
                "x": 0.5,
                "xanchor": "center",
                "y": 0.0,
                "yanchor": "bottom",
            },
            margin={**fig.layout.margin.to_plotly_json(), "t": 10, "b": 100},
        )

        dtype_label = str(series_dtype(y_test_series))
        content = [
            pn.pane.Markdown(f"**{target_name}** ({dtype_label})"),
            (target_stats, fig),
        ]
        return title, content

    @block
    def block_confusion_matrix(self, title: str = "") -> BlockContent:
        """Render confusion matrix for classification predictions.

        Parameters
        ----------
        title : str, default=""
            Optional section title.

        Returns
        -------
        tuple[str, list[pn.viewable.Viewable]]
            Section title and confusion matrix content rendered by the @block decorator.

        Examples
        --------
        >>> runtime.block_confusion_matrix()
        """
        explainer = self._require_explainer("confusion_matrix")
        if self.y_test is None or self.y_pred is None:
            raise ValueError("confusion_matrix block requires y_test and predicted values from the explainer.")
        y_test = cast(TargetValues, self.y_test)
        y_pred = cast(TargetValues, self.y_pred)
        fig = plot_confusion_matrix(y_true=y_test, y_pred=y_pred, colors_dict=explainer.colors_dict)
        if title is None:
            return "Confusion matrix", [fig]
        return title, [fig]

    @block
    def block_lift_curve(
        self,
        title: str = "",
        label: int | str = -1,
        selection: list[Any] | None = None,
        nb: int = 100,
        target_fraction: float = 0.1,
        max_points: int = 2000,
        width: int = 900,
        height: int = 600,
    ) -> BlockContent:
        """Render lift curve for classification probabilities.

        Parameters
        ----------
        title : str, default=""
            Optional section title.
        label : int or str, default=-1
            Class identifier used to select the target probability column.
        selection : list[Any] or None, default=None
            Optional subset of sample indices to include.
        nb : int, default=100
            Number of intervals used to build the curve.
        target_fraction : float, default=0.1
            Share of ranked population used to compute Lift@k.
        max_points : int, default=2000
            Maximum number of observations used by the plot.
        width : int, default=900
            Plot width in pixels.
        height : int, default=600
            Plot height in pixels.

        Returns
        -------
        tuple[str, list[pn.viewable.Viewable]]
            Section title and lift curve content rendered by the @block decorator.

        Examples
        --------
        >>> runtime.block_lift_curve()
        """
        explainer = self._require_explainer("lift_curve")

        if getattr(explainer, "_case", None) != "classification":
            raise ValueError("lift_curve block is only available for classification case.")

        if explainer.y_target is None:
            raise ValueError("lift_curve block requires target values on the explainer.")

        label_num, label_code, label_value = explainer.check_label_name(label)

        if explainer.proba_values is None:
            explainer.predict_proba()

        fig = plot_lift_curve(
            x_data=explainer.x_init,
            y_target=explainer.y_target,
            y_proba_values=explainer.proba_values,
            style_dict=explainer.plot._style_dict,
            selection=selection,
            label_num=label_num,
            label_code=label_code,
            label_value=label_value,
            nb=nb,
            target_fraction=target_fraction,
            max_points=max_points,
            width=width,
            height=height,
        )

        if title is None:
            return "Lift curve", [fig]
        return title, [fig]

    @block
    def block_univariate_analysis(
        self,
        title: str = "Univariate analysis",
        show_train: bool = True,
    ) -> BlockContent:
        """Render per-feature univariate analysis with interactive selection.

        Parameters
        ----------
        title : str, default="Univariate analysis"
            Section title displayed above selector and analysis panel.
        show_train : bool, default=True
            Whether train statistics are shown alongside prediction statistics.

        Returns
        -------
        tuple[str, list[pn.viewable.Viewable]]
            Section title and univariate analysis content rendered by the @block decorator.

        Examples
        --------
        >>> runtime.block_univariate_analysis()
        """
        df_train_test = self._require_train_test_data("univariate_analysis")
        explainer = self._require_explainer("univariate_analysis")

        df = df_train_test
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
            fig.update_layout(
                title={
                    **fig.layout.title.to_plotly_json(),
                    "x": 0.5,
                    "xanchor": "center",
                    "y": 0.0,
                    "yanchor": "bottom",
                },
                margin={**fig.layout.margin.to_plotly_json(), "t": 10, "b": 100},
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
                    _coerce_viewable(col_stats),
                    _coerce_viewable(fig),
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
            return title, [pn.pane.Markdown("No feature available.")]

        feature_select = pn.widgets.Select(
            name="Feature",
            options=list(feature_panels.keys()),
            value=next(iter(feature_panels)),
            sizing_mode="stretch_width",
        )
        selected_panel = pn.panel(pn.bind(cast(Any, lambda selected: feature_panels[selected]), feature_select))

        return title, [feature_select, selected_panel]

    def _preprocess_train_data(self, x_train: pd.DataFrame | None) -> pd.DataFrame | None:
        if x_train is None or self.explainer is None:
            return x_train
        x_train_pre = inverse_transform(x_train, self.explainer.preprocessing)
        x_train_pre = handle_categorical_missing(x_train_pre)
        if self.explainer.postprocessing:
            x_train_pre = apply_postprocessing(x_train_pre, self.explainer.postprocessing)
        return x_train_pre

    @staticmethod
    def _get_values_and_name(
        y: pd.DataFrame | pd.Series | list[Any] | None, default_name: str
    ) -> tuple[TargetValues | None, str | None]:
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

    def _require_train_test_data(self, block_type: str) -> pd.DataFrame:
        if self.df_train_test is None:
            raise ValueError(f"{block_type} block requires x_train and explainer.x_init data on the report instance.")
        return self.df_train_test

    def _feature_label(self, feature: str) -> str:
        if self.explainer is None:
            return feature
        return self.explainer.features_dict.get(feature, feature)

    def _feature_distribution_colors(self) -> dict:
        explainer = self._require_explainer("feature_distribution")
        return explainer.colors_dict["report_feature_distribution"]
