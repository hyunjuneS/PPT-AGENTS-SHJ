import logging
import os
from datetime import datetime
from pathlib import Path

PACKAGE_DIR = Path(__file__).parent.parent

LOGGING_LEVEL = int(os.getenv("DEEPPRESENTER_LOG_LEVEL", logging.INFO))
HEAVY_REFLECT = os.getenv("DEEPPRESENTER_HEAVY_REFLECT", "").lower() in ("1", "true", "yes")

RETRY_TIMES = int(os.getenv("RETRY_TIMES", 3))
TOOL_CUTOFF_LEN = int(os.getenv("TOOL_CUTOFF_LEN", 4096))
CONTEXT_LENGTH_LIMIT = int(os.getenv("CONTEXT_LENGTH_LIMIT", 200_000))
INSPECT_CONTENT_MAX_CALLS = int(os.getenv("INSPECT_CONTENT_MAX_CALLS", 3))

WORKSPACE_BASE = Path(
    os.getenv(
        "DEEPPRESENTER_WORKSPACE_BASE",
        str(PACKAGE_DIR / "output" / datetime.now().strftime("%Y%m%d")),
    )
)

GLOBAL_ENV_LIST = [
    "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "ALL_PROXY",
    "http_proxy", "https_proxy", "no_proxy", "all_proxy",
]

# ============ Agent Prompts ============

OFFLINE_PROMPT = """
<Offline Mode>
You are operating in offline mode without internet access.
</Offline Mode>
"""

CONTEXT_MODE_PROMPT = """
<Context Mode>
You are operating in limited working context. Save files and intermediate results immediately after generation.
</Context Mode>
"""

HALF_BUDGET_NOTICE_MSG = {
    "type": "text",
    "text": "<NOTICE>You have used about half of your working budget. Focus on the core task and skip unnecessary explorations.</NOTICE>",
}

URGENT_BUDGET_NOTICE_MSG = {
    "type": "text",
    "text": "<URGENT>Working budget nearly exhausted. Finish the core task and call `finalize` now.</URGENT>",
}

