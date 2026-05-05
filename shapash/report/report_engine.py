"""
report_engine.py — block-based report engine + legacy generation helpers.

Shapash Theme Version: White background, gold accents, and sidebar navigation.
"""

from __future__ import annotations

import importlib
import logging
import os
import re
from pathlib import Path

import pandas as pd
import papermill as pm
import yaml
from nbconvert import HTMLExporter

from shapash.utils.utils import get_project_root

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette shared by all built-in blocks (Updated for Shapash Light Theme)
# ─────────────────────────────────────────────────────────────────────────────

PALETTE = {
    "gold": {"bg": "#ffffff", "border": "#ffbb00", "title": "#ccac00", "text": "#333333"},
    "blue": {"bg": "#ffffff", "border": "#2255aa", "title": "#2255aa", "text": "#333333"},
    "gray": {"bg": "#ffffff", "border": "#eeeeee", "title": "#666666", "text": "#666666"},
    "orange": {"bg": "#fff9e6", "border": "#ffbb00", "title": "#cc8833", "text": "#444444"},
}


# ─────────────────────────────────────────────────────────────────────────────
# ReportBase
# ─────────────────────────────────────────────────────────────────────────────


class ReportBase:
    """
    Base class for block-based HTML reports.
    Methods named block_<type> return HTML strings for specific sections.
    """

    def render_block(self, block_cfg: dict) -> str:
        """Dispatch one YAML block entry to the matching block_* method."""
        block_type = block_cfg.get("type", "")
        params = block_cfg.get("params", {})
        method = getattr(self, f"block_{block_type}", None)

        if method is None:
            if block_type == "custom":
                return self._render_custom(block_cfg)
            logger.warning("Unknown block type '%s' — skipped.", block_type)
            return ""

        try:
            return method(**params)
        except Exception as exc:
            logger.error("Block '%s' raised: %s", block_type, exc)
            return _error_html(block_type, exc)

    def _render_custom(self, block_cfg: dict) -> str:
        """Call an arbitrary importable function."""
        func_path = block_cfg.get("function", "")
        params = block_cfg.get("params", {})
        try:
            mod_path, fn_name = func_path.rsplit(".", 1)
            fn = getattr(importlib.import_module(mod_path), fn_name)
            return fn(self, **params)
        except Exception as exc:
            logger.error("Custom block '%s' raised: %s", func_path, exc)
            return _error_html(func_path, exc)

    # ── Built-in blocks (Shapash Styled) ──────────────────────────────────────

    def block_header(self, title: str = "Report", subtitle: str = "") -> str:
        """Large page title with centered text."""
        sub = f'<div class="shapash-callout"><p>{subtitle}</p></div>' if subtitle else ""
        return f'<div class="main-header"><h1>{title}</h1>{sub}</div>'

    def block_text(
        self,
        title: str = "",
        body: str = "",
        color: str = "gray",
    ) -> str:
        """Standard section with a title and paragraph."""
        h2 = f'<h2 class="section-title">{title}</h2>' if title else ""
        return f'<div class="content-block">{h2}<p>{body}</p></div>'

    def block_key_value(
        self,
        title: str = "",
        items: dict | None = None,
        color: str = "gold",
    ) -> str:
        """Two-column key/value metadata table."""
        items = items or {}
        rows = "".join(f'<tr><td class="kv-key">{k} :</td><td class="kv-val">{v}</td></tr>' for k, v in items.items())
        h2 = f'<h2 class="section-title">{title}</h2>' if title else ""
        return f'<div class="content-block">{h2}<table class="kv-table">{rows}</table></div>'

    def block_badge_row(
        self,
        title: str = "",
        badges: list | None = None,
    ) -> str:
        """A row of metrics or badges."""
        badges = badges or []
        pills = ""
        for b in badges:
            c = PALETTE.get(b.get("color", "gray"), PALETTE["gray"])
            pills += (
                f'<span class="badge" style="border-color:{c["border"]}">'
                f'<span style="color:{c["title"]};font-weight:600">{b.get("label", "")}</span>'
                f'<span style="margin-left:8px">{b.get("value", "")}</span></span>'
            )
        h2 = f'<h2 class="section-title">{title}</h2>' if title else ""
        return f'<div class="content-block">{h2}<div style="display:flex;flex-wrap:wrap;gap:10px">{pills}</div></div>'

    def block_callout(
        self,
        body: str = "",
        color: str = "gold",
        icon: str = "",
    ) -> str:
        """The distinct Shapash left-border callout box."""
        return f'<div class="shapash-callout"><p>{body}</p></div>'

    def block_divider(self, label: str = "") -> str:
        """Thin light rule for separating sections."""
        return '<div class="shapash-divider"></div>'


# ─────────────────────────────────────────────────────────────────────────────
# New declarative pipeline
# ─────────────────────────────────────────────────────────────────────────────


def generate_report(
    report: ReportBase,
    config_file: str,
    output_file: str,
) -> None:
    """Render a ReportBase instance to an HTML file driven by a YAML config."""
    cfg_path = Path(config_file).resolve()
    print(f"Loading config → {cfg_path}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)

    sections = cfg.get("sections")
    if not sections:
        raise ValueError("YAML config must have a top-level 'sections' list.")

    logger.info("Rendering %d block(s)…", len(sections))

    rendered_blocks = []
    nav_links = []

    # Process each block and build the sidebar
    for block_cfg in sections:
        block_html = report.render_block(block_cfg)
        params = block_cfg.get("params", {})
        title = params.get("title")
        block_type = block_cfg.get("type")

        # If the block has a title (and isn't the main header), add it to the sidebar
        if title and block_type != "header":
            section_id = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")

            wrapped_html = f'<section id="{section_id}" class="scroll-section">{block_html}</section>'

            nav_links.append(f'<a class="nav-item" href="#{section_id}">{title}</a>')
        else:
            wrapped_html = f'<section class="scroll-section">{block_html}</section>'

        rendered_blocks.append(wrapped_html)

    # Join the navigation links into a single string
    sidebar_html = "\n".join(nav_links)

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logo_path = Path(__file__).resolve().parents[2] / "docs" / "assets" / "images" / "svg" / "shapash-github.svg"
    logo_src = os.path.relpath(logo_path, out_path.parent).replace(os.sep, "/") if logo_path.exists() else ""

    # Pass BOTH the body and the sidebar to the HTML template
    out_path.write_text(_html_page("\n".join(rendered_blocks), sidebar_html, logo_src), encoding="utf-8")
    logger.info("Report saved → %s", output_file)


# ─────────────────────────────────────────────────────────────────────────────
# Legacy pipeline
# ─────────────────────────────────────────────────────────────────────────────


def execute_report(
    working_dir: str,
    explainer: object,
    project_info_file: str,
    x_train: pd.DataFrame | None = None,
    y_train: pd.DataFrame | None = None,
    y_test: pd.Series | pd.DataFrame | None = None,
    config: dict | None = None,
    notebook_path: str | None = None,
    kernel_name: str | None = None,
) -> None:
    """Run the legacy notebook-based report generation pipeline.

    The function serializes the explainer and optional train/test datasets into
    ``working_dir``, then executes the report notebook with Papermill.
    """
    if config is None:
        config = {}
    explainer.save(path=os.path.join(working_dir, "smart_explainer.pickle"))
    if x_train is not None:
        x_train.to_csv(os.path.join(working_dir, "x_train.csv"))
    if y_train is not None:
        y_train.to_csv(os.path.join(working_dir, "y_train.csv"))
    if y_test is not None:
        y_test.to_csv(os.path.join(working_dir, "y_test.csv"))

    root_path = get_project_root()
    if not notebook_path:
        notebook_path = os.path.join(root_path, "shapash", "report", "base_report.ipynb")

    pm.execute_notebook(
        notebook_path,
        os.path.join(working_dir, "base_report.ipynb"),
        parameters=dict(dir_path=working_dir, project_info_file=project_info_file, config=config),
        kernel_name=kernel_name,
    )


def export_and_save_report(working_dir: str, output_file: str) -> None:
    """Export the executed legacy notebook in ``working_dir`` to an HTML file."""
    root_path = get_project_root()
    exporter = HTMLExporter(
        exclude_input=True,
        extra_template_basedirs=[os.path.join(root_path, "shapash", "report", "template")],
        template_name="custom",
        exclude_anchor_links=True,
    )
    body, _ = exporter.from_filename(filename=os.path.join(working_dir, "base_report.ipynb"))
    with open(output_file, "w") as f:
        f.write(body)


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers (Shapash UI Styling)
# ─────────────────────────────────────────────────────────────────────────────


def _error_html(block_id: str, exc: Exception) -> str:
    return (
        f'<div style="background:#fff1f1;border:1px solid #d9534f;padding:20px;border-radius:6px;margin:20px 0">'
        f'<h2 style="color:#d9534f;font-size:1rem">⚠ Block "{block_id}" failed</h2>'
        f'<pre style="color:#a94442;white-space:pre-wrap;font-size:12px">{exc}</pre></div>'
    )


def _html_page(body: str, sidebar_html: str = "", logo_src: str = "") -> str:
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

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Shapash Report</title>
  <style>
    :root {{ --shapash-gold: #ffbb00; --text-main: #333; --text-light: #777; }}
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    
    html {{ scroll-behavior: smooth; }}
    
    body {{ 
        background: #fdfdfd; color: var(--text-main); 
        font-family: 'Helvetica Neue', Arial, sans-serif; 
        display: flex; min-height: 100vh;
    }}
    
    .sidebar {{
        width: 240px; background: #fff; border-right: 1px solid #eee;
        position: fixed; height: 100vh; padding: 30px 20px;
        overflow-y: auto; 
    }}
    .sidebar-brand {{ 
        margin-bottom: 40px; display: flex; align-items: center; gap: 10px;
        color: var(--shapash-gold); font-size: 18px; font-weight: bold;
    }}
    .sidebar-brand-logo {{ display: block; width: 34px; height: 34px; flex: 0 0 auto; }}
    .sidebar-brand-text {{ line-height: 1.2; }}
    .nav-item {{ 
        color: var(--text-light); padding: 10px 0; display: block; 
        text-decoration: none; font-size: 13px; transition: 0.2s;
    }}
    .nav-item:hover {{ color: var(--text-main); }}
    .nav-item.active {{ 
        color: #551a8b; font-weight: bold; border-left: 3px solid #551a8b; padding-left: 10px; 
    }}

    /* FIX 1: Changed bottom padding to 60vh (60% of viewport height) to add scroll room */
    .container {{ margin-left: 240px; width: 100%; padding: 60px 80px 60vh; max-width: 1200px; }}
    
    .main-header {{ text-align: center; margin-bottom: 60px; }}
    .main-header h1 {{ font-size: 2.4rem; font-weight: 500; color: #000; margin-bottom: 20px; }}
    .section-title {{ font-size: 1.6rem; color: #000; margin: 40px 0 20px; font-weight: 500; }}
    .content-block {{ margin-bottom: 30px; line-height: 1.6; font-size: 14px; }}
    .shapash-callout {{ border-left: 4px solid var(--shapash-gold); background: #fff; padding: 15px 25px; margin: 30px 0; color: #333; line-height: 1.6; font-size: 15px; }}
    .kv-table {{ width: 100%; border-collapse: collapse; }}
    .kv-key {{ font-weight: bold; width: 140px; padding: 8px 0; vertical-align: top; color: #000; }}
    .kv-val {{ padding: 8px 0; color: var(--text-main); }}
    .badge {{ display: inline-block; padding: 6px 14px; border: 1px solid #eee; border-radius: 4px; font-size: 12px; background: #fff; }}
    .shapash-divider {{ border-bottom: 1px solid #eee; margin: 50px 0; }}

    .scroll-section {{ scroll-margin-top: 40px; }}

    @media (max-width: 900px) {{
        .sidebar {{ display: none; }}
        .container {{ margin-left: 0; padding: 30px 40vh; }}
    }}
  </style>
</head>
<body>
  <nav class="sidebar">
        {brand_html}
    {sidebar_html}
  </nav>
  <main class="container">
    {body}
  </main>
  
  <script>
    document.addEventListener('DOMContentLoaded', function() {{
        const sections = document.querySelectorAll('.scroll-section[id]');
        const navItems = document.querySelectorAll('.nav-item');

        function onScroll() {{
            let currentId = '';
            
            sections.forEach(section => {{
                const sectionTop = section.offsetTop;
                // Trigger slightly earlier as we scroll down
                if (window.scrollY >= (sectionTop - 150)) {{
                    currentId = section.getAttribute('id');
                }}
            }});

            // FIX 2: If we are at the absolute bottom of the page, force the last section to be active
            if ((window.innerHeight + window.scrollY) >= document.body.offsetHeight - 5) {{
                if (sections.length > 0) {{
                    currentId = sections[sections.length - 1].getAttribute('id');
                }}
            }}

            navItems.forEach(item => {{
                item.classList.remove('active');
                if (item.getAttribute('href') === '#' + currentId) {{
                    item.classList.add('active');
                }}
            }});
        }}
        
        window.addEventListener('scroll', onScroll);
        onScroll(); // Trigger once on load
    }});
  </script>
</body>
</html>"""
