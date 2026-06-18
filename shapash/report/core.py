"""Smart report orchestration for block-based HTML reports."""

from __future__ import annotations

import base64
import html
import logging
import re
from pathlib import Path

import panel as pn

from shapash.report.panel_support import apply_report_css, report_js_text
from shapash.report.validation import load_report_config

logger = logging.getLogger(__name__)


def generate_report(runtime, config_file: str, output_file: str) -> None:
    """Render a Panel report to an HTML file driven by a YAML config."""
    cfg_path = Path(config_file).resolve()
    cfg = load_report_config(cfg_path)
    print(f"Loading config → {cfg_path}")

    _assign_section_ids(cfg["sections"])

    rendered_blocks = [runtime.render_block(block_cfg) for block_cfg in cfg["sections"]]
    nav_bar = build_navigation_bar(cfg["sections"])

    out_path = Path(output_file).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    apply_report_css(custom_css=cfg.get("custom_css"), base_dir=cfg_path.parent)
    report_content = pn.Column(
        *[block for block in rendered_blocks if block is not None],
        css_classes=["report-content"],
        sizing_mode="stretch_width",
    )
    report_layout = pn.Row(
        pn.Column(nav_bar, css_classes=["report-sidebar"], width=300, sizing_mode="fixed"),
        report_content,
        css_classes=["main-report"],
        sizing_mode="stretch_width",
    )
    report_layout.append(pn.pane.HTML(f"<script>{report_js_text()}</script>", sizing_mode="stretch_width"))
    report_layout.save(str(out_path), embed=True, resources="cdn")
    logger.info("Report saved → %s", output_file)


def _slugify(text: str) -> str:
    """Return a stable slug for navigation anchor IDs."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug


def _block_label(block_cfg: dict) -> str:
    """Resolve a human-readable block label for the navigation bar."""
    params = block_cfg.get("params", {})
    if isinstance(params, dict):
        title = params.get("title")
    else:
        title = ""
    if isinstance(title, str) and title.strip():
        return title.strip()
    block_type = block_cfg.get("type", "section")
    label = str(block_type).replace("_", " ").strip().title()
    if label:
        return label
    return "Section"


def _assign_section_ids(blocks: list[dict], used: set[str] | None = None, prefix: str = "section") -> None:
    """Assign unique anchor IDs to all blocks (including group children)."""
    used_ids = used if used is not None else set()
    for idx, block in enumerate(blocks, start=1):
        label_slug = _slugify(_block_label(block))
        if label_slug:
            base = label_slug
        else:
            base = f"{prefix}-{idx}"
        candidate = base
        suffix = 2
        while candidate in used_ids:
            candidate = f"{base}-{suffix}"
            suffix += 1
        used_ids.add(candidate)
        block["_section_id"] = candidate
        if block.get("type") == "group":
            children = block.get("blocks", [])
            if isinstance(children, list):
                _assign_section_ids(children, used=used_ids, prefix=f"{candidate}-item")


def _wrap_section_anchor(content: pn.viewable.Viewable, section_id: str | None) -> pn.Column:
    """Wrap one rendered block with an in-page anchor target."""
    if not section_id:
        return pn.Column(content, css_classes=["scroll-section"], sizing_mode="stretch_width")
    anchor = pn.pane.HTML(f'<div id="{section_id}" class="scroll-anchor"></div>', sizing_mode="stretch_width")
    return pn.Column(anchor, content, css_classes=["scroll-section"], sizing_mode="stretch_width")


def build_navigation_bar(blocks: list[dict]) -> pn.pane.HTML:
    """Build a sticky in-page navigation bar using Panel HTML pane."""
    items_html: list[str] = []
    item_count = 0
    for block in blocks:
        block_type = block.get("type")
        label = html.escape(_block_label(block))
        section_id = html.escape(str(block.get("_section_id", "")))
        if block_type == "group":
            item_count += 1
            children_links: list[str] = []
            for child in block.get("blocks", []):
                child_label = html.escape(_block_label(child))
                child_id = html.escape(str(child.get("_section_id", "")))
                item_count += 1
                children_links.append(f'<a class="nav-item nav-child" href="#{child_id}">{child_label}</a>')
            items_html.append(
                "".join(
                    [
                        '<div class="nav-group">',
                        f'<a class="nav-item nav-group-title" href="#{section_id}">{label}</a>',
                        '<div class="nav-group-children">',
                        *children_links,
                        "</div>",
                        "</div>",
                    ]
                )
            )
            continue

        item_count += 1
        items_html.append(f'<a class="nav-item" href="#{section_id}">{label}</a>')

    logo_path = Path(__file__).resolve().parent.parent / "style" / "shapash-fond-clair.png"
    logo_data = base64.b64encode(logo_path.read_bytes()).decode("ascii")
    logo_html = '<div class="nav-logo">' f'<img src="data:image/png;base64,{logo_data}" alt="Shapash logo" />' "</div>"

    nav_scale = max(0.62, min(1.0, 24 / max(1, item_count)))
    nav_html = "".join(
        [
            f'<nav class="report-nav" style="--nav-scale: {nav_scale:.3f};">',
            logo_html,
            '<div class="nav-current" aria-live="polite">',
            '<span class="nav-current-label">You are here</span>',
            '<span class="nav-current-value">Top of report</span>',
            "</div>",
            *items_html,
            "</nav>",
        ]
    )
    return pn.pane.HTML(nav_html, sizing_mode="stretch_width")
