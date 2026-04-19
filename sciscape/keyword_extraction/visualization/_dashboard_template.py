"""HTML template for the keyword extraction dashboard.

The actual template lives in ``dashboard_template.html`` alongside this
module.  This file loads it once at import time so that existing code
can continue to reference ``_DASHBOARD_HTML_TEMPLATE`` unchanged.
"""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).with_name("dashboard_template.html")
_DASHBOARD_HTML_TEMPLATE: str = _TEMPLATE_PATH.read_text(encoding="utf-8")
