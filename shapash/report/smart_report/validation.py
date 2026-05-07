"""Validation and lightweight utility helpers for HTML reports."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml


def load_report_config(cfg_path: Path) -> dict:
    """Load and validate a report YAML configuration file."""
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    try:
        with cfg_path.open(encoding="utf-8") as file:
            cfg = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML syntax in '{cfg_path}': {exc}") from exc

    validate_report_schema(cfg, cfg_path)
    return cfg


def validate_report_schema(cfg: object, cfg_path: Path) -> None:
    """Validate the minimal schema expected by the report renderer."""
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML structure in '{cfg_path}': top-level content must be a mapping.")

    sections = cfg.get("sections")
    if not isinstance(sections, list) or not sections:
        raise ValueError(f"Invalid YAML structure in '{cfg_path}': 'sections' must be a non-empty list.")

    for idx, block in enumerate(sections, start=1):
        _validate_block(block, idx, cfg_path)


def _validate_block(block: object, idx: int, cfg_path: Path, parent: str = "sections") -> None:
    if not isinstance(block, dict):
        raise ValueError(f"Invalid YAML structure in '{cfg_path}': {parent}[{idx}] must be a mapping.")

    block_type = block.get("type")
    if not isinstance(block_type, str) or not block_type.strip():
        raise ValueError(f"Invalid YAML structure in '{cfg_path}': {parent}[{idx}].type must be a non-empty string.")

    params = block.get("params", {})
    if not isinstance(params, dict):
        raise ValueError(f"Invalid YAML structure in '{cfg_path}': {parent}[{idx}].params must be a mapping.")

    if block_type == "custom":
        function_path = block.get("function")
        if not isinstance(function_path, str) or not function_path.strip():
            raise ValueError(
                f"Invalid YAML structure in '{cfg_path}': {parent}[{idx}].function is required for custom blocks."
            )

    if block_type == "group":
        child_blocks = block.get("blocks", [])
        if not isinstance(child_blocks, list):
            raise ValueError(
                f"Invalid YAML structure in '{cfg_path}': {parent}[{idx}].blocks must be a list for group blocks."
            )
        for child_idx, child_block in enumerate(child_blocks, start=1):
            _validate_block(child_block, child_idx, cfg_path, parent=f"{parent}[{idx}].blocks")


def render_block_error_html(block_id: str, exc: Exception) -> str:
    """Render a consistent HTML error box for block failures."""
    return (
        f'<div style="background:#fff1f1;border:1px solid #d9534f;padding:20px;border-radius:6px;margin:20px 0">'
        f'<h2 style="color:#d9534f;font-size:1rem">⚠ Block "{block_id}" failed</h2>'
        f'<pre style="color:#a94442;white-space:pre-wrap;font-size:12px">{exc}</pre></div>'
    )


def stats_to_table(test_stats: dict, names: list[str], train_stats: dict | None = None) -> pd.DataFrame:
    """Build a stats table and drop columns that are entirely missing."""
    if train_stats is not None:
        stats_table = pd.DataFrame({names[1]: pd.Series(train_stats), names[0]: pd.Series(test_stats)})
    else:
        stats_table = pd.DataFrame({names[0]: pd.Series(test_stats)})

    return stats_table.dropna(axis=1, how="all")
