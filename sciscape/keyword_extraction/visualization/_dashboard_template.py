"""HTML template for the keyword extraction dashboard.

The actual template lives in ``dashboard_template.html`` alongside this
module.  This file loads it once at import time so that existing code
can continue to reference ``_DASHBOARD_HTML_TEMPLATE`` unchanged.
"""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).with_name("dashboard_template.html")
if _TEMPLATE_PATH.exists():
    _DASHBOARD_HTML_TEMPLATE: str = _TEMPLATE_PATH.read_text(encoding="utf-8")
else:
    import warnings
    warnings.warn(
        f"Dashboard template not found: {_TEMPLATE_PATH}. "
        "Report generation will fail. Reinstall the package.",
        stacklevel=1,
    )
    _DASHBOARD_HTML_TEMPLATE: str = "<html><body><h1>Template missing</h1></body></html>"
