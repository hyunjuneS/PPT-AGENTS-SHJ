"""Headroom (https://github.com/headroomlabs-ai/headroom) tool-output compression
for the LangGraph engine.

Applied only to the model-facing view of the conversation, exactly like
engine.cap_images(): graph state keeps every ToolMessage byte-for-byte, so
finalize detection, the budget-notice injection and the saved .history dumps of
graph state are unaffected. Only what actually goes over the wire to the model
on a given call is compressed.

What gets compressed: headroom's SmartCrusher, via
headroom.integrations.langchain.compress_tool_messages. In practice it only
rewrites large JSON arrays of similar records (e.g. a 200-row result becomes one
schema line plus CSV-like rows); plain text, Markdown, HTML, code and logs come
back unchanged (was_modified=False), so the HTML read -> edit_file flow keeps
seeing exact strings.

Safety rails on top of headroom's own (min size, error-content preservation):
  * Only str-content ToolMessages are passed in. List content (e.g. inspect_slide
    results carrying an image block) is never touched — headroom would otherwise
    str() the list and send the image dict as text.
  * Tools in HEADROOM_EXCLUDE_TOOLS (default: read_file, finalize) are skipped.
    read_file is excluded because the model may copy exact substrings out of a
    file it read (JSON included) into edit_file's old_string.
  * The compressed message is a model_copy of the original, so name/status/
    tool_call_id/additional_kwargs survive (headroom's own rebuild drops them).
  * Any failure (import or runtime) falls back to the uncompressed messages.

Environment:
  HEADROOM_COMPRESSION      on/off (default on; no-op if headroom-ai isn't installed)
  HEADROOM_MIN_TOKENS       min estimated tokens (chars/4) before a ToolMessage
                            is considered (default 500)
  HEADROOM_EXCLUDE_TOOLS    comma-separated tool names never compressed
                            (default "read_file,finalize")

headroom's anonymous telemetry beacon is already off by default (opt-in); it is
pinned off here as well, since this app runs on an internal network.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from langchain_core.messages import BaseMessage, ToolMessage

logger = logging.getLogger(__name__)

_FALSY = ("0", "false", "no", "off")

COMPRESSION_ENABLED = os.getenv("HEADROOM_COMPRESSION", "on").lower().strip() not in _FALSY
MIN_TOKENS_TO_COMPRESS = int(os.getenv("HEADROOM_MIN_TOKENS", "500"))
EXCLUDED_TOOLS = frozenset(
    t.strip() for t in os.getenv("HEADROOM_EXCLUDE_TOOLS", "read_file,finalize").split(",") if t.strip()
)

_compress_fn = None
_unavailable = False


def _load():
    """Lazy import: headroom pulls in litellm (~3s on first import), so it is only
    paid on the first model call that actually has a ToolMessage to consider."""
    global _compress_fn, _unavailable
    if _compress_fn is None and not _unavailable:
        os.environ.setdefault("HEADROOM_TELEMETRY", "off")
        os.environ.setdefault("HEADROOM_BEACON", "off")
        try:
            from headroom.integrations.langchain import compress_tool_messages

            _compress_fn = compress_tool_messages
        except Exception as e:  # ImportError, or a broken optional dep inside headroom
            _unavailable = True
            logger.warning("Headroom compression disabled — could not import headroom-ai: %s", e)
    return _compress_fn


@dataclass
class CompressionStats:
    messages_compressed: int = 0
    chars_before: int = 0
    chars_after: int = 0

    @property
    def chars_saved(self) -> int:
        return self.chars_before - self.chars_after

    def as_record(self) -> dict:
        return {
            "headroom_messages_compressed": self.messages_compressed,
            "headroom_chars_saved": self.chars_saved,
            # Same chars/4 heuristic headroom itself uses for its thresholds — an
            # estimate, not the backend tokenizer's count (input_tokens is the real one).
            "headroom_est_tokens_saved": self.chars_saved // 4,
        }


def _tool_names_by_id(messages: list[BaseMessage]) -> dict[str, str]:
    names: dict[str, str] = {}
    for m in messages:
        for tc in getattr(m, "tool_calls", None) or ():
            if tc.get("id") and tc.get("name"):
                names[tc["id"]] = tc["name"]
    return names


def compress_for_model(messages: list[BaseMessage]) -> tuple[list[BaseMessage], CompressionStats]:
    """Return (model-facing messages, stats). Never mutates the input list or its messages."""
    stats = CompressionStats()
    if not COMPRESSION_ENABLED:
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

    out = list(messages)
    for i, compressed in zip(candidates, result.messages):
        original = messages[i]
        if compressed is original or compressed.content == original.content:
            continue
        out[i] = original.model_copy(update={"content": compressed.content})
        stats.messages_compressed += 1
        stats.chars_before += len(original.content)
        stats.chars_after += len(compressed.content)

    if stats.messages_compressed:
        logger.info(
            "Headroom: compressed %d tool message(s), %d -> %d chars",
            stats.messages_compressed, stats.chars_before, stats.chars_after,
        )
    return out, stats
