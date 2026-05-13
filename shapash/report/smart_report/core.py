"""Smart report orchestration for block-based HTML reports."""

from __future__ import annotations

import importlib
import logging
from pathlib import Path

import pandas as pd

from shapash.report.smart_report.blocks import ReportBlockMixin
from shapash.report.smart_report.layout import build_html_page, render_sections, resolve_logo_src
from shapash.report.smart_report.validation import load_report_config, render_block_error_html

logger = logging.getLogger(__name__)


def create_block_runtime(
    explainer=None,
    x_train: pd.DataFrame | None = None,
    y_train: pd.Series | pd.DataFrame | list | None = None,
    y_test: pd.Series | pd.DataFrame | list | None = None,
    config: dict | None = None,
    block_class: type[ReportBlockMixin] | None = None,
):
    """Create a runtime object that holds report state and block methods."""
    runtime_cls = type("BlockRuntime", (block_class or ReportBlockMixin,), {})
    runtime = runtime_cls()

    runtime.explainer = explainer
    runtime.config = config or {}
    runtime.x_train_init = x_train
    runtime.x_train_pre = runtime._preprocess_train_data(x_train)
    runtime.x_init = getattr(explainer, "x_init", None)
    runtime.df_train_test = runtime._create_train_test_df(test=runtime.x_init, train=runtime.x_train_pre)
    runtime.y_train, runtime.target_name_train = runtime._get_values_and_name(y_train, "target")
    runtime.y_test, runtime.target_name_test = runtime._get_values_and_name(y_test, "target")
    runtime.target_name = runtime.target_name_train or runtime.target_name_test
    runtime.max_points = runtime.config.get("max_points", 200)

    if explainer is not None:
        if explainer.y_pred is not None:
            runtime.y_pred, _ = runtime._get_values_and_name(explainer.y_pred, "prediction")
        else:
            runtime.y_pred = explainer.model.predict(explainer.x_encoded)
    else:
        runtime.y_pred = None

    return runtime


def generate_report(runtime, config_file: str, output_file: str) -> None:
    """Render a report to an HTML file driven by a YAML config."""
    cfg_path = Path(config_file).resolve()
    cfg = load_report_config(cfg_path)
    print(f"Loading config → {cfg_path}")

    runtime.render_block = lambda block_cfg: render_block(runtime, block_cfg)
    rendered_blocks, sidebar_html = render_sections(runtime, cfg["sections"])

    out_path = Path(output_file).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    body_html = "\n".join(rendered_blocks)
    logo_src = resolve_logo_src(out_path.parent)
    out_path.write_text(build_html_page(body=body_html, sidebar_html=sidebar_html, logo_src=logo_src), encoding="utf-8")
    logger.info("Report saved → %s", output_file)


def render_block(runtime, block_cfg: dict) -> str:
    """Dispatch one YAML block entry to the matching block_* method."""
    block_type = block_cfg.get("type", "")
    params = block_cfg.get("params", {})

    if block_type == "group":
        return "".join(render_block(runtime, child_cfg) for child_cfg in block_cfg.get("blocks", []))

    method = getattr(runtime, f"block_{block_type}", None)
    if method is None:
        if block_type == "custom":
            return _render_custom(runtime, block_cfg)
        logger.warning("Unknown block type '%s' — skipped.", block_type)
        return ""

    try:
        return method(**params)
    except Exception as exc:
        logger.error("Block '%s' raised: %s", block_type, exc)
        return render_block_error_html(block_type, exc)


def _render_custom(runtime, block_cfg: dict) -> str:
    """Call an arbitrary importable function."""
    func_path = block_cfg.get("function", "")
    params = block_cfg.get("params", {})
    try:
        mod_path, fn_name = func_path.rsplit(".", 1)
        fn = getattr(importlib.import_module(mod_path), fn_name)
        return fn(runtime, **params)
    except Exception as exc:
        logger.error("Custom block '%s' raised: %s", func_path, exc)
        return render_block_error_html(func_path, exc)
