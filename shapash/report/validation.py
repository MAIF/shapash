"""Validate report YAML configuration and provide rendering helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import panel as pn
import yaml


def load_report_config(cfg_path: Path) -> dict[str, Any]:
    """Load a YAML report configuration from disk and validate its structure.

    Parameters
    ----------
    cfg_path : Path
        Path to the YAML configuration file.

    Returns
    -------
    dict[str, Any]
        The validated top-level configuration mapping.

    Raises
    ------
    FileNotFoundError
        If ``cfg_path`` does not exist.
    ValueError
        If the file contains invalid YAML or does not match the required schema.
    """
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
    """Validate the required top-level structure of a report configuration.

    The configuration must be a mapping with a non-empty ``sections`` list.
    Each section is checked recursively for a non-empty ``type`` string, a
    mapping-valued ``params`` field, and any fields required by its block type.

    Parameters
    ----------
    cfg : object
        Parsed YAML content to validate. It is accepted as ``object`` because
        YAML may produce values of any type before validation.
    cfg_path : Path
        Source path used to identify the configuration in error messages.

    Raises
    ------
    ValueError
        If the top-level content or any section has an invalid structure.
    """
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML structure in '{cfg_path}': top-level content must be a mapping.")

    sections = cfg.get("sections")
    if not isinstance(sections, list) or not sections:
        raise ValueError(f"Invalid YAML structure in '{cfg_path}': 'sections' must be a non-empty list.")

    for idx, block in enumerate(sections, start=1):
        _validate_block(block, idx, cfg_path)


def _validate_block(block: object, idx: int, cfg_path: Path, parent: str = "sections") -> None:
    """Validate a single report block and recursively validate group children.

    Parameters
    ----------
    block : object
        Parsed block configuration to validate.
    idx : int
        One-based position of the block in its containing list.
    cfg_path : Path
        Source path used to identify the configuration in error messages.
    parent : str, default="sections"
        Location of the containing list, used to build a precise error path.

    Raises
    ------
    ValueError
        If the block is not a mapping, has invalid parameters, is missing the
        function path required by a custom block, or has malformed children.
    """
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


def render_block_error(block_id: str, exc: Exception) -> pn.pane.Alert:
    """Create a Panel alert describing a report block failure.

    Parameters
    ----------
    block_id : str
        Identifier of the block that failed.
    exc : Exception
        Exception raised while rendering the block.

    Returns
    -------
    pn.pane.Alert
        A danger-styled alert containing the block identifier and exception.
    """
    return pn.pane.Alert(
        f'Block "{block_id}" failed\n\n{exc}',
        alert_type="danger",
        sizing_mode="stretch_width",
    )


def stats_to_table(
    test_stats: dict[str, Any], names: list[str], train_stats: dict[str, Any] | None = None
) -> pd.DataFrame:
    """Build a statistics table and remove columns containing only missing values.

    Parameters
    ----------
    test_stats : dict[str, Any]
        Test-set statistics, keyed by statistic name.
    names : list[str]
        Column labels. ``names[0]`` labels test statistics and, when
        ``train_stats`` is provided, ``names[1]`` labels training statistics.
    train_stats : dict[str, Any] or None, default=None
        Optional training-set statistics, keyed by statistic name.

    Returns
    -------
    pandas.DataFrame
        Statistics arranged in columns, with all-missing columns removed.
    """
    if train_stats is not None:
        stats_table = pd.DataFrame({names[1]: pd.Series(train_stats), names[0]: pd.Series(test_stats)})
    else:
        stats_table = pd.DataFrame({names[0]: pd.Series(test_stats)})

    return stats_table.dropna(axis=1, how="all")
