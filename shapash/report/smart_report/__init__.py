"""Public entry point for block-based HTML smart reports."""

from shapash.report.smart_report.blocks import PALETTE
from shapash.report.smart_report.core import create_block_runtime, generate_report, render_block

__all__ = ["PALETTE", "create_block_runtime", "generate_report", "render_block"]
