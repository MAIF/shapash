"""Panel helpers for standalone smart report HTML rendering."""

from __future__ import annotations

import contextlib
import io
import re
from functools import lru_cache

import panel as pn
from panel.io.resources import CDN_DIST, Resources


def render_plotly_pane_html(fig) -> str:
    """Render a Plotly figure as a standalone Panel fragment."""
    _enable_panel_plotly()
    pane = pn.pane.Plotly(fig, config={"responsive": True})
    # Panel may write empty lines to stdout/stderr while building mimebundle.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        bundle = pane._repr_mimebundle_()
    data = bundle[0] if isinstance(bundle, tuple) else bundle
    html = data.get("text/html")
    if not html:
        raise ValueError("Panel Plotly pane did not return HTML output.")
    return f'<div class="panel-plot">{html}</div>'

# @lru_cache is used to avoid redundant computation of resource tags across multiple panes.
@lru_cache(maxsize=1)
def panel_resource_tags() -> str:
    """Return the CSS and JS tags required to hydrate Panel panes."""
    _enable_panel_plotly()
    resources = Resources(mode="cdn")
    css_html = _normalize_panel_css(resources.render_css())
    js_html = resources.render_js() if callable(resources.render_js) else resources.render_js
    js_html = _ensure_panel_runtime(js_html)
    return "\n".join(part for part in [css_html, js_html] if part)

@lru_cache(maxsize=1)
def _enable_panel_plotly() -> None:
    pn.extension("plotly")


def _normalize_panel_css(css_html: str) -> str:
    return re.sub(
        r'href="static/extensions/panel/([^"?]+)(?:\?v=[^"]+)?',
        lambda match: f'href="{CDN_DIST}{match.group(1)}',
        css_html,
    )


def _ensure_panel_runtime(js_html: str) -> str:
    if "panel.min.js" in js_html:
        return js_html

    panel_tag = f'<script type="text/javascript" src="{CDN_DIST}panel.min.js"></script>'
    marker = '<script type="text/javascript">\n  Bokeh.set_log_level("info");\n</script>'
    if marker in js_html:
        return js_html.replace(marker, f"{panel_tag}\n\n{marker}")
    return f"{js_html}\n{panel_tag}"
