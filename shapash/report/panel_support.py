"""Panel helpers for smart report rendering."""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

import panel as pn


@lru_cache(maxsize=1)
def _enable_panel_plotly() -> None:
    """Enable Plotly support in Panel extension, cached to run only once per session."""
    pn.extension("plotly")


@lru_cache(maxsize=1)
def report_js_text() -> str:
    """Load report JavaScript once for Panel report export."""
    js_path = Path(__file__).resolve().parent / "report_script.js"
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
    _enable_panel_plotly()
    css_paths = [Path(__file__).resolve().parent / "report_styles.css"]
    css_paths.extend(_resolve_custom_css_paths(custom_css=custom_css, base_dir=base_dir))

    for css_path in css_paths:
        css = css_path.read_text(encoding="utf-8")
        if css not in pn.config.raw_css:
            pn.config.raw_css.append(css)
