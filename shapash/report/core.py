"""Smart report orchestration for block-based HTML reports."""

from __future__ import annotations

import html
import importlib
import logging
import re
from pathlib import Path

import pandas as pd
import panel as pn

from shapash.report.blocks import ReportBlockMixin
from shapash.report.panel_support import apply_report_css, report_js_text
from shapash.report.validation import load_report_config, render_block_error

logger = logging.getLogger(__name__)


def create_block_runtime(
    explainer=None,
    x_train: pd.DataFrame | None = None,
    y_train: pd.Series | pd.DataFrame | list | None = None,
    y_test: pd.Series | pd.DataFrame | list | None = None,
    config: dict | None = None,
    block_instance: ReportBlockMixin | None = None,
):
    """Create a runtime object that holds report state and block methods."""
    if block_instance is None:
        runtime = ReportBlockMixin()
    else:
        runtime = block_instance

    runtime.explainer = explainer
    if config is None:
        runtime.config = {}
    else:
        runtime.config = config
    runtime.x_train_init = x_train
    runtime.x_train_pre = runtime._preprocess_train_data(x_train)
    runtime.x_init = getattr(explainer, "x_init", None)
    runtime.df_train_test = runtime._create_train_test_df(test=runtime.x_init, train=runtime.x_train_pre)
    runtime.y_train, runtime.target_name_train = runtime._get_values_and_name(y_train, "target")
    runtime.y_test, runtime.target_name_test = runtime._get_values_and_name(y_test, "target")
    if runtime.target_name_train is not None:
        runtime.target_name = runtime.target_name_train
    else:
        runtime.target_name = runtime.target_name_test
    runtime.max_points = runtime.config.get("max_points", 200)
    runtime._inside_group = False

    if explainer is not None:
        if explainer.y_pred is not None:
            runtime.y_pred, _ = runtime._get_values_and_name(explainer.y_pred, "prediction")
        else:
            runtime.y_pred = explainer.model.predict(explainer.x_encoded)
    else:
        runtime.y_pred = None

    return runtime


def generate_report(runtime, config_file: str, output_file: str) -> None:
    """Render a Panel report to an HTML file driven by a YAML config."""
    cfg_path = Path(config_file).resolve()
    cfg = load_report_config(cfg_path)
    print(f"Loading config → {cfg_path}")

    _assign_section_ids(cfg["sections"])

    runtime.render_block = lambda block_cfg: render_block(runtime, block_cfg)
    rendered_blocks = [render_block(runtime, block_cfg) for block_cfg in cfg["sections"]]
    nav_bar = build_navigation_bar(cfg["sections"])

    out_path = Path(output_file).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    apply_report_css()
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


def render_block(runtime, block_cfg: dict):
    """Dispatch one YAML block entry to the matching block_* method."""
    block_type = block_cfg.get("type", "")
    params = block_cfg.get("params", {})

    if block_type == "group":
        previous_inside_group = getattr(runtime, "_inside_group", False)
        runtime._inside_group = True
        try:
            children = [render_block(runtime, child_cfg) for child_cfg in block_cfg.get("blocks", [])]
        finally:
            runtime._inside_group = previous_inside_group
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

    method = getattr(runtime, f"block_{block_type}", None)
    if method is None:
        if block_type == "custom":
            return _render_custom(runtime, block_cfg)
        logger.warning("Unknown block type '%s' — skipped.", block_type)
        return None

    try:
        result = method(**params)
        if isinstance(result, pn.viewable.Viewable):
            return _wrap_section_anchor(result, block_cfg.get("_section_id"))
        if isinstance(result, str):
            return _wrap_section_anchor(pn.pane.Markdown(result), block_cfg.get("_section_id"))
        return _wrap_section_anchor(pn.panel(result), block_cfg.get("_section_id"))
    except Exception as exc:
        logger.error("Block '%s' raised: %s", block_type, exc)
        return render_block_error(block_type, exc)


def _render_custom(runtime, block_cfg: dict):
    """Call an arbitrary importable function."""
    func_path = block_cfg.get("function", "")
    params = block_cfg.get("params", {})
    try:
        mod_path, fn_name = func_path.rsplit(".", 1)
        fn = getattr(importlib.import_module(mod_path), fn_name)
        result = fn(runtime, **params)
        if isinstance(result, pn.viewable.Viewable):
            return result
        if isinstance(result, str):
            return pn.pane.Markdown(result)
        return pn.panel(result)
    except Exception as exc:
        logger.error("Custom block '%s' raised: %s", func_path, exc)
        return render_block_error(func_path, exc)


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

    nav_scale = max(0.62, min(1.0, 24 / max(1, item_count)))
    nav_html = "".join(
        [
            f'<nav class="report-nav" style="--nav-scale: {nav_scale:.3f};">',
            '<div class="nav-current" aria-live="polite">',
            '<span class="nav-current-label">You are here</span>',
            '<span class="nav-current-value">Top of report</span>',
            "</div>",
            *items_html,
            "</nav>",
        ]
    )
    return pn.pane.HTML(nav_html, sizing_mode="stretch_width")
