"""Panel helpers for smart report rendering."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
import panel as pn
import plotly.graph_objs as go

# Use Tabulator only for wide tables; smaller tables keep the simpler DataFrame pane rendering.
TABULATOR_MIN_COLUMNS = 10


def report_js_text() -> str:
    """Load report JavaScript once for Panel report export."""
    js_path = Path(__file__).resolve().parent / "assets" / "report_script.js"
    return js_path.read_text(encoding="utf-8")


def _resolve_custom_css_paths(
    custom_css: str | Path | Iterable[str | Path] | None,
    base_dir: str | Path | None,
) -> list[Path]:
    if custom_css is None:
        return []

    values: Iterable[str | Path]
    if isinstance(custom_css, str | Path):
        values = [custom_css]
    else:
        values = custom_css

    resolved: list[Path] = []
    base_path = Path(base_dir).resolve() if base_dir is not None else None
    for value in values:
        css_path = Path(value)
        if not css_path.is_absolute() and base_path is not None:
            css_path = base_path / css_path
        css_path = css_path.resolve()
        if not css_path.exists():
            raise FileNotFoundError(f"Custom CSS file not found: {css_path}")
        if css_path.suffix.lower() != ".css":
            raise ValueError(f"Custom CSS file must use .css extension: {css_path}")
        resolved.append(css_path)
    return resolved


def apply_report_css(
    custom_css: str | Path | Iterable[str | Path] | None = None,
    base_dir: str | Path | None = None,
) -> None:
    """Register smart-report CSS in Panel global configuration."""
    css_paths = [Path(__file__).resolve().parent / "assets" / "report_styles.css"]
    css_paths.extend(_resolve_custom_css_paths(custom_css=custom_css, base_dir=base_dir))

    for css_path in css_paths:
        css = css_path.read_text(encoding="utf-8")
        if css not in pn.config.raw_css:
            pn.config.raw_css.append(css)


def _dedupe_css_classes(*class_groups: Any) -> list[str]:
    classes: list[str] = []
    for group in class_groups:
        if not group:
            continue
        if isinstance(group, str):
            items = [group]
        else:
            items = list(group)
        for item in items:
            if item and item not in classes:
                classes.append(item)
    return classes


def _add_css_classes(viewable: pn.viewable.Viewable, *classes: str) -> pn.viewable.Viewable:
    current = getattr(viewable, "css_classes", None)
    merged = _dedupe_css_classes(current, classes)
    if merged:
        viewable.css_classes = merged
    return viewable


def _auto_style_viewable(viewable: Any, method_name: str | None = None) -> Any:
    if isinstance(viewable, pn.pane.Markdown):
        return _add_css_classes(viewable, "content-block")

    if isinstance(viewable, pn.pane.DataFrame):
        classes = ["kv-table"]
        if getattr(viewable, "width_policy", None) == "min":
            classes.append("fit-content-table")
        return _add_css_classes(viewable, *classes)

    if isinstance(viewable, pn.pane.Plotly):
        return viewable

    if isinstance(viewable, pn.widgets.Select):
        return viewable

    tabulator_type = getattr(pn.widgets, "Tabulator", None)
    if tabulator_type is not None and isinstance(viewable, tabulator_type):
        return _add_css_classes(viewable, "kv-table")

    if isinstance(viewable, pn.Spacer):
        return viewable

    param_function_type = getattr(pn.param, "ParamFunction", None)
    if param_function_type is not None and isinstance(viewable, param_function_type):
        return viewable

    param_method_type = getattr(pn.param, "ParamMethod", None)
    if param_method_type is not None and isinstance(viewable, param_method_type):
        return viewable

    if isinstance(viewable, pn.Row):
        if method_name == "block_badge_row":
            for child in getattr(viewable, "objects", []):
                if isinstance(child, pn.pane.Markdown):
                    _add_css_classes(child, "badge-pill")
        return viewable

    if isinstance(viewable, pn.Column):
        if method_name == "block_project_information":
            _add_css_classes(viewable, "project-info-grid")
            for child in getattr(viewable, "objects", []):
                if isinstance(child, pn.Column):
                    _add_css_classes(child, "project-info-card")
                    for grandchild in getattr(child, "objects", []):
                        _auto_style_viewable(grandchild, method_name=method_name)
                else:
                    _auto_style_viewable(child, method_name=method_name)
            return viewable

        for child in getattr(viewable, "objects", []):
            _auto_style_viewable(child, method_name=method_name)
        return viewable

    method_info = f" in '{method_name}'" if method_name else ""
    allowed_types = "Markdown, DataFrame, Plotly, Select, Tabulator, Spacer, ParamFunction, ParamMethod, Row, Column"
    raise TypeError(
        f"Unsupported Panel object type returned{method_info}: {type(viewable).__name__}. "
        f"Allowed Panel return types: {allowed_types}."
    )


def _coerce_viewable(item: Any) -> pn.viewable.Viewable:
    if isinstance(item, pn.viewable.Viewable):
        return item
    if isinstance(item, str):
        return pn.pane.Markdown(item)
    if isinstance(item, pd.DataFrame):
        tabulator_type = getattr(pn.widgets, "Tabulator", None)
        if tabulator_type is not None and item.shape[1] > TABULATOR_MIN_COLUMNS:
            return tabulator_type(
                item,
                disabled=True,
                show_index=False,
                layout="fit_columns",
                width_policy="max",
                sizing_mode="stretch_width",
            )
        return pn.pane.DataFrame(item, index=False, width_policy="min", sizing_mode="stretch_width")
    if isinstance(item, go.Figure):
        return pn.pane.Plotly(item, config={"responsive": True}, sizing_mode="stretch_width")
    raise TypeError(
        f"Unsupported block return type: {type(item).__name__}. "
        "Supported types: strings, pandas DataFrame, Plotly Figures, Panel Viewable."
    )
