import os
import yaml
from pathlib import Path
from report_engine import ReportBase, generate_report, PALETTE

# ── Path logic ───────────────────────────────────────────────────────────────
# This finds the absolute path to the directory containing demo.py
HERE = Path(__file__).resolve().parent

# Define absolute paths for all files
config_in = HERE / "report_config.yml"
config_out_ext = HERE / "report_config_extended.yml"
report_base_out = HERE / "report_base.html"
report_ext_out = HERE / "report_extended.html"

# ─────────────────────────────────────────────────────────────────────────────
# Example 1 — use ReportBase as-is
# ─────────────────────────────────────────────────────────────────────────────

base_report = ReportBase()
# Convert Path objects to strings for the engine
generate_report(base_report, str(config_in), str(report_base_out))
print(f"✅  Saved: {report_base_out}")


# ─────────────────────────────────────────────────────────────────────────────
# Example 2 — subclass to add a new block type
# ─────────────────────────────────────────────────────────────────────────────


class ExtendedReport(ReportBase):
    def block_progress_bar(self, title: str = "", items: list | None = None, color: str = "blue") -> str:
        items = items or []
        c = PALETTE.get(color, PALETTE["blue"])
        bars = ""
        for item in items:
            pct = max(0, min(100, int(item.get("pct", 0))))
            bars += f"""
            <div style="margin-bottom:10px">
                <div style="display:flex;justify-content:space-between;
                            color:{c["text"]};font-size:12px;margin-bottom:3px">
                    <span>{item.get("label", "")}</span>
                    <span>{pct}%</span>
                </div>
                <div style="background:{c["bg"]};border:1px solid {c["border"]};
                            border-radius:4px;height:10px;overflow:hidden">
                    <div style="width:{pct}%;height:100%;
                                background:{c["border"]};border-radius:4px">
                    </div>
                </div>
            </div>"""
        h2 = f'<h2 class="block-title" style="color:{c["title"]}">{title}</h2>' if title else ""
        return f'<div class="block" style="background:{c["bg"]};border-left:4px solid {c["border"]}">{h2}{bars}</div>'

    def block_pie_chart(self, title: str = "", data: dict | None = None, color: str = "blue") -> str:
        return f'<div class="block"><h2>{title}</h2><p>Tarte à la crème</p></div>'


extended_report = ExtendedReport()
generate_report(extended_report, str(config_out_ext), str(report_ext_out))
print(f"✅  Saved: {report_ext_out} (+ block_progress_bar)")
