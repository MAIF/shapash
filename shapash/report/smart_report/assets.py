"""Static CSS and JavaScript fragments for HTML report rendering."""

from pathlib import Path

_ASSETS_DIR = Path(__file__).resolve().parent
_STYLE_FILE = _ASSETS_DIR / "report_styles.css"
_SCRIPT_FILE = _ASSETS_DIR / "report_script.js"

REPORT_STYLES = _STYLE_FILE.read_text(encoding="utf-8")
REPORT_SCRIPT = f"<script>\n{_SCRIPT_FILE.read_text(encoding='utf-8')}\n</script>"
