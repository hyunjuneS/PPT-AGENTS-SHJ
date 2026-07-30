import functools
import inspect
import logging
import time
import traceback
from typing import Any

from deeppresenter.utils.constants import LOGGING_LEVEL

# Single named logger for the whole deeppresenter package. Propagates to the
# root logger (configured once, in main-ui.py's logging.basicConfig) so every
# log line — deeppresenter's, httpx's, uvicorn's — shares one format and one
# stream instead of each module keeping its own handler/format.
logger = logging.getLogger("deeppresenter")
logger.setLevel(LOGGING_LEVEL)


def debug(msg, *args, **kwargs): logger.debug(msg, *args, **kwargs)
def info(msg, *args, **kwargs): logger.info(msg, *args, **kwargs)
def warning(msg, *args, **kwargs): logger.warning(msg, *args, **kwargs)
def error(msg, *args, **kwargs): logger.error(msg, *args, **kwargs)


# ── Pretty progress display ───────────────────────────────────────────────────

_R = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_CYAN = "\033[36m"
_YELLOW = "\033[33m"
_GREEN = "\033[32m"
_RED = "\033[31m"
_MAGENTA = "\033[35m"

_AGENT_ICON = {"Research": "🔬", "Design": "🎨", "Planner": "📋"}
_TOOL_ICON = {
    "write_file": "✍ ", "read_file": "📖", "inspect_slide": "🔍",
    "inspect_manuscript": "🔍", "inspect_content": "🔍", "execute_command": "⚡", "finalize": "🏁",
    "web_search": "🌐", "web_fetch": "🌐",
}


def _shorten(s: str, n: int = 60) -> str:
    s = s.strip().replace("\n", " ")
    return "…" + s[-n:] if len(s) > n else s


def show_agent_start(name: str, max_turns: int | None = None) -> None:
    icon = _AGENT_ICON.get(name, "🤖")
    limit = f"  (max {max_turns} turns)" if max_turns else ""
    print(f"\n{_BOLD}{_CYAN}{icon}  {name} Agent{_R}{_DIM}{limit}{_R}", flush=True)


def show_agent_turn(name: str, turn: int, max_turns: int | None = None) -> None:
    limit = f"/{max_turns}" if max_turns else ""
    print(f"  {_DIM}── turn {turn}{limit} {'─' * 36}{_R}", flush=True)


def show_tool_call(tool: str, args: dict) -> None:
    icon = _TOOL_ICON.get(tool, "⚙ ")
    # Pick the most informative argument to display
    arg_val = ""
    for key in ("path", "html_file", "outcome", "command", "query", "url"):
        if key in args:
            arg_val = _shorten(str(args[key]))
            break
    arg_str = f"  {_DIM}{arg_val}{_R}" if arg_val else ""
    print(f"    {_YELLOW}{icon} {tool}{_R}{arg_str}", flush=True)


def show_tool_result(text: str, is_error: bool = False) -> None:
    color = _RED if is_error else _GREEN
    icon = "✗" if is_error else "✓"
    print(f"      {color}{icon}{_R} {_DIM}{_shorten(text, 80)}{_R}", flush=True)


def show_agent_done(name: str, turns: int, elapsed: float) -> None:
    icon = _AGENT_ICON.get(name, "🤖")
    print(
        f"  {_GREEN}✓{_R} {_BOLD}{name}{_R} done"
        f"  {_DIM}{turns} turns  {elapsed:.1f}s{_R}\n",
        flush=True,
    )


class timer:
    def __init__(self, name: str = ""):
        self.name = name

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, *_):
        elapsed = time.time() - self._start
        if elapsed > 1:
            debug(f"{self.name} took {elapsed:.2f}s")

    def __call__(self, func):
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                start = time.time()
                try:
                    return await func(*args, **kwargs)
                finally:
                    elapsed = time.time() - start
                    if elapsed > 1:
                        debug(f"{self.name or func.__name__} took {elapsed:.2f}s")
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                start = time.time()
                try:
                    return func(*args, **kwargs)
                finally:
                    elapsed = time.time() - start
                    if elapsed > 1:
                        debug(f"{self.name or func.__name__} took {elapsed:.2f}s")
            return sync_wrapper


def logging_openai_exceptions(identifier: Any, exc: Exception) -> str:
    msg = f"Exception [{type(exc).__name__}]: {str(exc)}\n{traceback.format_exc()}"
    warning(f"{identifier} → {msg}")
    return msg
