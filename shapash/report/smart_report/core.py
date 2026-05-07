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


class ReportBase(ReportBlockMixin):
    """Base class for block-based HTML reports."""

    def __init__(
        self,
        explainer=None,
        x_train: pd.DataFrame | None = None,
        y_train: pd.Series | pd.DataFrame | list | None = None,
        y_test: pd.Series | pd.DataFrame | list | None = None,
        config: dict | None = None,
    ):
        self.explainer = explainer
        self.config = config or {}
        self.x_train_init = x_train
        self.x_train_pre = self._preprocess_train_data(x_train)
        self.x_init = getattr(explainer, "x_init", None)
        self.df_train_test = self._create_train_test_df(test=self.x_init, train=self.x_train_pre)
        self.y_train, self.target_name_train = self._get_values_and_name(y_train, "target")
        self.y_test, self.target_name_test = self._get_values_and_name(y_test, "target")
        self.target_name = self.target_name_train or self.target_name_test
        self.max_points = self.config.get("max_points", 200)

        if explainer is not None:
            if explainer.y_pred is not None:
                self.y_pred, _ = self._get_values_and_name(explainer.y_pred, "prediction")
            else:
                self.y_pred = explainer.model.predict(explainer.x_encoded)
        else:
            self.y_pred = None

    def generate_report(self, config_file: str, output_file: str) -> None:
        """Render a report instance to an HTML file driven by a YAML config."""
        cfg_path = Path(config_file).resolve()
        cfg = load_report_config(cfg_path)
        print(f"Loading config → {cfg_path}")

        rendered_blocks, sidebar_html = render_sections(self, cfg["sections"])

        out_path = Path(output_file).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        body_html = "\n".join(rendered_blocks)
        logo_src = resolve_logo_src(out_path.parent)
        out_path.write_text(
            build_html_page(body=body_html, sidebar_html=sidebar_html, logo_src=logo_src), encoding="utf-8"
        )
        logger.info("Report saved → %s", output_file)

    def render_block(self, block_cfg: dict) -> str:
        """Dispatch one YAML block entry to the matching block_* method."""
        block_type = block_cfg.get("type", "")
        params = block_cfg.get("params", {})

        if block_type == "group":
            return "".join(self.render_block(child_cfg) for child_cfg in block_cfg.get("blocks", []))

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
            return render_block_error_html(block_type, exc)

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
            return render_block_error_html(func_path, exc)
