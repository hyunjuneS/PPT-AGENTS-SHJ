"""Headroom (https://github.com/headroomlabs-ai/headroom) tool-output compression,
run in-process with the headroom-ai Python library right before every LangGraph
LLM call.

Applied only to the model-facing view of the conversation, exactly like
engine.cap_images(): graph state keeps every ToolMessage byte-for-byte, so
finalize detection, the budget-notice injection and the saved .history dumps of
graph state are unaffected. Only what actually goes over the wire to the model
on a given call is compressed.

What gets compressed: headroom's SmartCrusher, via
headroom.integrations.langchain.compress_tool_messages. In practice it only
rewrites large JSON arrays of similar records (e.g. a 200-row result becomes one
schema line plus CSV-like rows); plain text, Markdown, HTML, code and logs come
back unchanged, so the HTML read -> edit_file flow keeps seeing exact strings.

Safety rails on top of headroom's own (min size, error-content preservation):
  * Only str-content ToolMessages are passed in. List content (e.g. inspect_slide
    results carrying an image block) is never touched.
  * Tools in HEADROOM_EXCLUDE_TOOLS (default: read_file, finalize) are skipped.
    read_file is excluded because the model may copy exact substrings out of a
    file it read into edit_file's old_string.
  * The compressed message is a model_copy of the original, so name/status/
    tool_call_id/additional_kwargs survive.
  * Any failure (import or runtime) falls back to the uncompressed messages.

Every call that saved tokens is also appended to Headroom's own savings ledger
(~/.headroom/savings_events.jsonl, or HEADROOM_SAVINGS_EVENTS_PATH), so
`headroom savings` shows this app's totals next to any other Headroom usage.

Environment:
  HEADROOM_ENABLED          true/false (default false)
  HEADROOM_MIN_TOKENS       min estimated tokens (chars/4) before a ToolMessage
                            is considered (default 500)
  HEADROOM_EXCLUDE_TOOLS    comma-separated tool names never compressed
                            (default "read_file,finalize")

This app runs on an internal network, so before headroom is imported its
air-gap switch (HEADROOM_OFFLINE=1: no beacon, update check or model downloads)
and litellm's bundled price list (LITELLM_LOCAL_MODEL_COST_MAP=True: no GitHub
fetch on import) are turned on by default. Either can still be overridden by
setting it explicitly in the environment.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from langchain_core.messages import BaseMessage, ToolMessage

logger = logging.getLogger(__name__)

HEADROOM_ENABLED = os.getenv("HEADROOM_ENABLED", "false").lower().strip() in ("1", "true", "yes", "on")
MIN_TOKENS_TO_COMPRESS = int(os.getenv("HEADROOM_MIN_TOKENS", "500"))
EXCLUDED_TOOLS = frozenset(
    t.strip() for t in os.getenv("HEADROOM_EXCLUDE_TOOLS", "read_file,finalize").split(",") if t.strip()
)

_LEDGER_SOURCE = "ppt-agent"

_compress_fn = None
_record_fn = None
_unavailable = False


def _load():
    """Import headroom once. It pulls in litellm (a few seconds), so main-ui.py calls
    warm_up() at server start instead of paying it inside the first request."""
    global _compress_fn, _record_fn, _unavailable
    if _compress_fn is None and not _unavailable:
        os.environ.setdefault("HEADROOM_OFFLINE", "1")
        os.environ.setdefault("HEADROOM_TELEMETRY", "off")
        os.environ.setdefault("HEADROOM_BEACON", "off")
        os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        try:
            from headroom.integrations.langchain import compress_tool_messages

            _compress_fn = compress_tool_messages
        except Exception as e:  # ImportError, or a broken optional dep inside headroom
            _unavailable = True
            logger.warning("Headroom compression disabled — could not import headroom-ai: %s", e)
            return None
        try:
            from headroom.savings_ledger import record_savings_event

            _record_fn = record_savings_event
        except Exception as e:
            logger.warning("Headroom savings ledger unavailable, compression still on: %s", e)
    return _compress_fn


def warm_up() -> bool:
    """Import headroom ahead of time when HEADROOM_ENABLED. Returns True if it is usable."""
    return HEADROOM_ENABLED and _load() is not None


@dataclass
class CompressionStats:
    messages_compressed: int = 0
    chars_before: int = 0
    chars_after: int = 0
    tokens_before: int = 0
    tokens_after: int = 0

    @property
    def tokens_saved(self) -> int:
        return self.tokens_before - self.tokens_after

    def as_record(self) -> dict:
        return {
            "headroom_messages_compressed": self.messages_compressed,
            "headroom_chars_saved": self.chars_before - self.chars_after,
            # headroom's own token estimate, not the backend tokenizer's count
            # (input_tokens in the same record is the real one).
            "headroom_est_tokens_saved": self.tokens_saved,
        }


def _tool_names_by_id(messages: list[BaseMessage]) -> dict[str, str]:
    names: dict[str, str] = {}
    for m in messages:
        for tc in getattr(m, "tool_calls", None) or ():
            if tc.get("id") and tc.get("name"):
                names[tc["id"]] = tc["name"]
    return names


def compress_for_model(messages: list[BaseMessage], model: str | None = None) -> tuple[list[BaseMessage], CompressionStats]:
    """Return (model-facing messages, stats). Never mutates the input list or its messages."""
    stats = CompressionStats()
    if not HEADROOM_ENABLED:
        return messages, stats

    names = _tool_names_by_id(messages)
    candidates: list[int] = []
    for i, m in enumerate(messages):
        if not isinstance(m, ToolMessage) or not isinstance(m.content, str):
            continue
        tool_name = m.name or names.get(m.tool_call_id)
        if tool_name in EXCLUDED_TOOLS:
            continue
        if len(m.content) // 4 < MIN_TOKENS_TO_COMPRESS:
            continue
        candidates.append(i)

    if not candidates:
        return messages, stats

    compress = _load()
    if compress is None:
        return messages, stats

    try:
        result = compress([messages[i] for i in candidates], min_tokens_to_compress=MIN_TOKENS_TO_COMPRESS)
    except Exception as e:
        logger.warning("Headroom compression failed, sending uncompressed messages: %s", e)
        return messages, stats

    metrics_by_id = {mt.tool_call_id: mt for mt in result.metrics if mt.was_compressed}
    out = list(messages)
    for i, compressed in zip(candidates, result.messages):
        original = messages[i]
        if compressed is original or compressed.content == original.content:
            continue
        out[i] = original.model_copy(update={"content": compressed.content})
        stats.messages_compressed += 1
        stats.chars_before += len(original.content)
        stats.chars_after += len(compressed.content)
        mt = metrics_by_id.get(original.tool_call_id)
        if mt is not None:
            stats.tokens_before += mt.tokens_before
            stats.tokens_after += mt.tokens_after
        else:
            stats.tokens_before += len(original.content) // 4
            stats.tokens_after += len(compressed.content) // 4

    if stats.messages_compressed:
        logger.info(
            "Headroom: compressed %d tool message(s), %d -> %d tokens (est.)",
            stats.messages_compressed, stats.tokens_before, stats.tokens_after,
        )
        if _record_fn is not None:
            # Never raises (headroom's contract); one ledger line per LLM call.
            _record_fn(
                tokens_before=stats.tokens_before,
                tokens_after=stats.tokens_after,
                model=model,
                client=_LEDGER_SOURCE,
                source=_LEDGER_SOURCE,
            )
    return out, stats
