"""Render a Dash component tree to static HTML — for viewing Dash markup outside a running app.

A Dash component's ``str()``/``repr()`` is Python source (``Div([Span(...)])``), not markup: Dash
never turns components into HTML in Python — the JS renderer builds the DOM straight from that
Python tree over the wire. ``IPython.display.HTML(component)`` therefore just shows that repr as
literal text. :func:`dash_component_to_html` walks the same tree and emits actual markup instead,
and :class:`DashHtmlPreview` wraps it with a ``_repr_html_`` so Jupyter renders it automatically.
"""

from __future__ import annotations

import re
from html import escape
from typing import Any

_CAMEL_TO_KEBAB = re.compile(r"(?<!^)(?=[A-Z])")


def _style_to_css(style: dict) -> str:
    return ";".join(f"{_CAMEL_TO_KEBAB.sub('-', k).lower()}:{v}" for k, v in style.items())


def dash_component_to_html(component: Any) -> str:
    """Recursively render a Dash ``html.*`` component tree to an HTML string.

    Handles the plain attributes shapash's own components use: ``style`` (a dict), ``title``,
    ``className``. It does not handle interactive components (``dcc.Graph``, ``dcc.Dropdown``,
    callbacks...) — those have no meaning outside a running Dash app.

    Parameters
    ----------
    component : Any
        A Dash ``html.*`` component, a list of them, or a plain string/number (as found in
        ``children``).

    Returns
    -------
    str
        HTML markup equivalent to what Dash's JS renderer would mount for this tree.
    """
    if component is None:
        return ""
    if isinstance(component, (list, tuple)):
        return "".join(dash_component_to_html(c) for c in component)
    if isinstance(component, (str, int, float)):
        return escape(str(component))

    tag = type(component).__name__.lower()  # Div -> div, Span -> span, ...
    attrs = []
    style = getattr(component, "style", None)
    if style:
        attrs.append(f'style="{_style_to_css(style)}"')
    title = getattr(component, "title", None)
    if title:
        attrs.append(f'title="{escape(str(title))}"')
    class_name = getattr(component, "className", None)
    if class_name:
        attrs.append(f'class="{escape(str(class_name))}"')
    attr_str = "" if not attrs else " " + " ".join(attrs)

    inner = dash_component_to_html(getattr(component, "children", None))
    return f"<{tag}{attr_str}>{inner}</{tag}>"


class DashHtmlPreview:
    """Wraps a Dash component so Jupyter renders it as static HTML.

    ``explanation.plot.sentence(0)`` returns one of these: leave it as the last expression in a
    notebook cell (or ``display()`` it) and Jupyter calls ``_repr_html_`` automatically. The
    underlying Dash component — e.g. to embed in your own app's ``layout`` — is available as
    ``.component``.

    Panel components like ``plot_sentence_highlight`` are block-level ``<div>``s with no width of
    their own, because inside the webapp they should fill their panel. Outside one — e.g. a
    notebook's full output width — that same div stretches edge to edge regardless of how short
    the text is, leaving a wide slab of background colour next to it. This wraps the markup in an
    ``inline-block`` container so it hugs its content's width instead, capped at ``max_width`` so a
    genuinely long one still wraps within the cell rather than overflowing it.

    These components also never set their own default text colour on their light backgrounds —
    they rely on the *ambient* page text colour, which is safe inside the webapp because its
    Bootstrap page always supplies a light background with dark default text, but not in a
    notebook: a dark IDE theme (e.g. VS Code's default dark Jupyter theme) supplies a light-grey
    default text colour meant for a dark background, which is nearly invisible against these pale
    spans. This wrapper pins an explicit light background and dark text colour so the component
    renders the same regardless of the surrounding theme.
    """

    def __init__(self, component: Any, max_width: str = "100%") -> None:
        self.component = component
        self.max_width = max_width

    def _repr_html_(self) -> str:
        inner = dash_component_to_html(self.component)
        style = f"display:inline-block;max-width:{self.max_width};background-color:#ffffff;color:#212529"
        return f'<div style="{style}">{inner}</div>'

    def __repr__(self) -> str:
        return repr(self.component)
