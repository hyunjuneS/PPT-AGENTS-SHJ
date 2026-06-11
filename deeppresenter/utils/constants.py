import logging
import os
from datetime import datetime
from pathlib import Path

PACKAGE_DIR = Path(__file__).parent.parent

LOGGING_LEVEL = int(os.getenv("DEEPPRESENTER_LOG_LEVEL", logging.INFO))
MAX_LOGGING_LENGTH = int(os.getenv("DEEPPRESENTER_MAX_LOGGING_LENGTH", 1024))
HEAVY_REFLECT = os.getenv("DEEPPRESENTER_HEAVY_REFLECT", "").lower() in ("1", "true", "yes")

RETRY_TIMES = int(os.getenv("RETRY_TIMES", 3))
MAX_TOOLCALL_PER_TURN = int(os.getenv("MAX_TOOLCALL_PER_TURN", 7))
TOOL_CUTOFF_LEN = int(os.getenv("TOOL_CUTOFF_LEN", 4096))
CONTEXT_LENGTH_LIMIT = int(os.getenv("CONTEXT_LENGTH_LIMIT", 200_000))

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

AGENT_PROMPT = """
<Environment>
Current time: {time}
Working directory: {workspace}
Platform: Linux
Pre-installed tools: Python, curl, wget and other common utilities
Max tool calls per turn: {max_toolcall_per_turn}
Tool output exceeding {cutoff_len} characters will be truncated.
</Environment>

<Task Guidelines>
- Every response must include reasoning and a valid tool call.
- All tool calls in a single turn are processed in parallel — do not emit interdependent calls together.
</Task Guidelines>
"""

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

HIST_LOST_MSG = {
    "type": "text",
    "text": "<NOTICE>History between this point and the following message has been compacted into a summary</NOTICE>",
}

CONTINUE_MSG = {
    "type": "text",
    "text": "<NOTICE>History has been compacted. Refer to the saved summary and continue your work</NOTICE>",
}

LAST_ITER_MSG = {
    "type": "text",
    "text": "<URGENT>Working budget nearly exhausted. Call `finalize` now.</URGENT>",
}

MEMORY_COMPACT_MSG = """
You have reached the context length limit. Extract key information from tool interactions, generate a complete state summary, and save it to the working directory.

<summary_requirements>
1. Collected Information & Data
2. Uncertainties & Open Issues
3. Generated Artifacts (paths + purpose)
4. Next Steps (completed work, remaining tasks)
5. Lessons Learned
</summary_requirements>

Use {language} as the primary language. Save the summary directly to the working directory.
"""

MA_RESEACHER_PROMPT = ""
MA_RRESENTER_PROMPT = ""
