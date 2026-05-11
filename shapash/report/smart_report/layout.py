"""HTML layout and section rendering helpers for smart reports."""

from __future__ import annotations

import os
import re
from pathlib import Path

from shapash.report.smart_report.panel_support import panel_resource_tags

_ASSETS_DIR = Path(__file__).resolve().parent
_STYLE_FILE = _ASSETS_DIR / "report_styles.css"
_SCRIPT_FILE = _ASSETS_DIR / "report_script.js"

REPORT_STYLES = _STYLE_FILE.read_text(encoding="utf-8")
REPORT_SCRIPT = f"<script>\n{_SCRIPT_FILE.read_text(encoding='utf-8')}\n</script>"


def resolve_logo_src(base_dir: Path | None) -> str:
    """Resolve the relative path to the bundled Shapash logo."""
    if base_dir is None:
        return ""
    logo_path = Path(__file__).resolve().parents[3] / "docs" / "assets" / "images" / "svg" / "shapash-github.svg"
    return os.path.relpath(logo_path, base_dir).replace(os.sep, "/") if logo_path.exists() else ""


def block_title(block_cfg: dict) -> str:
    """Return the configured title for a block, if any."""
    return block_cfg.get("params", {}).get("title", "") or ""


def section_id(title: str) -> str:
    """Create a stable HTML id for a block title."""
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def wrap_section(block_html: str, html_id: str) -> str:
    """Wrap rendered HTML in a scroll-trackable section tag."""
    return f'<section id="{html_id}" class="scroll-section">{block_html}</section>'


def build_nav_link(title: str, html_id: str, extra_class: str = "") -> str:
    """Build a sidebar navigation link."""
    classes = " ".join(part for part in ["nav-item", extra_class] if part)
    return f'<a class="{classes}" href="#{html_id}">{title}</a>'


def render_block_section(report, block_cfg: dict) -> tuple[str, str | None]:
    """Render one non-group block and optionally wrap it as a scroll section."""
    block_html = report.render_block(block_cfg)
    title = block_title(block_cfg)
    if title and block_cfg.get("type") != "header":
        html_id = section_id(title)
        return wrap_section(block_html, html_id), html_id
    return block_html, None


def render_section_tree(report, block_cfg: dict, depth: int = 0) -> tuple[list[str], list[str]]:
    """Render one block tree (block or nested group) and its navigation entries."""
    if block_cfg.get("type") != "group":
        block_html, html_id = render_block_section(report, block_cfg)
        nav_links: list[str] = []
        if html_id:
            extra_class = f"nav-child nav-level-{depth}" if depth > 0 else ""
            nav_links.append(build_nav_link(block_title(block_cfg), html_id, extra_class=extra_class))
        return [block_html], nav_links

    rendered_blocks: list[str] = []
    child_nav_links: list[str] = []
    for child_cfg in block_cfg.get("blocks", []):
        child_blocks, child_nav = render_section_tree(report, child_cfg, depth=depth + 1)
        rendered_blocks.extend(child_blocks)
        child_nav_links.extend(child_nav)

    group_title = block_title(block_cfg)
    if not group_title:
        return rendered_blocks, child_nav_links

    group_id = section_id(group_title)
    if depth == 0:
        title_class = "nav-group-title"
        children_class = "nav-children"
    else:
        title_class = f"nav-child nav-group-title nav-level-{depth}"
        children_class = f"nav-children nav-level-{depth + 1}"

    nav_html = (
        '<div class="nav-group">'
        f'{build_nav_link(group_title, group_id, extra_class=title_class)}'
        f'<div class="{children_class}">{"".join(child_nav_links)}</div>'
        "</div>"
    )
    return [wrap_section("", group_id), *rendered_blocks], [nav_html]


def render_sections(report, sections: list[dict]) -> tuple[list[str], str]:
    """Render all configured sections and build the sidebar navigation HTML."""
    rendered_blocks: list[str] = []
    nav_links: list[str] = []

    for block_cfg in sections:
        block_tree, block_nav = render_section_tree(report, block_cfg)
        rendered_blocks.extend(block_tree)
        nav_links.extend(block_nav)

    return rendered_blocks, "\n".join(nav_links)


def build_sidebar_fragment(sidebar_html: str = "", logo_src: str = "") -> str:
    """Render the sidebar brand and navigation markup."""
    brand_html = (
        (
            f'<div class="sidebar-brand">'
            f'<img class="sidebar-brand-logo" src="{logo_src}" alt="Shapash logo">'
            f'<span class="sidebar-brand-text">Shapash Report</span>'
            f"</div>"
        )
        if logo_src
        else '<div class="sidebar-brand">Shapash Report</div>'
    )
    return f"{brand_html}{sidebar_html}"


def build_report_fragment(body: str, sidebar_html: str = "", logo_src: str = "") -> str:
    """Compose the styled report shell body fragment."""
    return f"""
    <style>{REPORT_STYLES}</style>
    <div class="report-shell">
        <nav class="sidebar">{build_sidebar_fragment(sidebar_html, logo_src)}</nav>
        <main class="container">
            {body}
        </main>
    </div>
    {REPORT_SCRIPT}
"""


def build_html_page(body: str, sidebar_html: str = "", logo_src: str = "") -> str:
    """Compose the full HTML page for a rendered report."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Shapash Report</title>
    {panel_resource_tags()}
</head>
<body>
  {build_report_fragment(body, sidebar_html, logo_src)}
</body>
</html>"""
