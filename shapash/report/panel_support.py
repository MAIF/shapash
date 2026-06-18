"""Panel helpers for smart report rendering."""

from __future__ import annotations

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


def apply_report_css() -> None:
    """Register smart-report CSS in Panel global configuration."""
    _enable_panel_plotly()
    css_path = Path(__file__).resolve().parent / "report_styles.css"
    css = css_path.read_text(encoding="utf-8")
    if css not in pn.config.raw_css:
        pn.config.raw_css.append(css)


def make_plotly_pane(fig) -> pn.pane.Plotly:
    """Build a responsive Plotly pane for report blocks."""
    _enable_panel_plotly()
    return pn.pane.Plotly(fig, config={"responsive": True}, sizing_mode="stretch_width")
