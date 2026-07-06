import logging

from deeppresenter.utils.config import LLM, get_json_from_response
from deeppresenter.utils.typings import ChatMessage, Role

logger = logging.getLogger(__name__)

DEFAULT_MIN_PAGES = 6
DEFAULT_MAX_PAGES = 30
_MAX_DOC_CHARS = 40_000  # cap document content sent for this decision call only

_SYSTEM_PROMPT = """You are a presentation-planning assistant. Your ONLY job is to
read the provided source document and decide how many slides a presentation
manuscript generated from it should contain. You do not write any slide content
yourself, and you must not ask questions. Respond with STRICT JSON ONLY — no
markdown code fences, no commentary before or after — matching exactly this shape:
{"num_pages": <integer>, "reasoning": "<one short sentence>"}"""

_USER_PROMPT_TEMPLATE = """Analyze the source document and the user's instruction below,
then decide an appropriate total slide count for the presentation manuscript that
will be generated from this document.

Consider:
- The depth, breadth, and structural complexity of the document content (number of
distinct topics/sections, amount of data/evidence, number of logical steps in the
narrative). Base this on the actual content below, not just the instruction text.
- The user's instruction, if it hints at a desired scope, audience, or length.
- The slide count MUST include the cover slide (slide 1) and the closing/summary
slide (the last slide) — it is a total count, not a count of "body" slides only.
- Prefer a range of 8-20 slides for typical documents. Only go outside that range
if the document is unusually short/simple or unusually long/dense. In any case,
the final number must be between {min_pages} and {max_pages}.

<user_instruction>
{instruction}
</user_instruction>

<source_document>
{document}
</source_document>

Respond with STRICT JSON ONLY, exactly in this shape and nothing else:
{{"num_pages": <integer between {min_pages} and {max_pages}>, "reasoning": "<one sentence>"}}"""


def _truncate_document(markdown_document: str) -> str:
    doc = markdown_document.strip()
    if len(doc) <= _MAX_DOC_CHARS:
        return doc
    return doc[:_MAX_DOC_CHARS] + f"\n\n... (truncated, original length {len(doc)} characters)"


async def decide_num_pages(
    llm: LLM,
    markdown_document: str,
    instruction: str,
    min_pages: int = DEFAULT_MIN_PAGES,
    max_pages: int = DEFAULT_MAX_PAGES,
) -> int:
    """Read markdown_document + instruction, ask llm for a slide count, clamp and return it.

    Falls back to the midpoint of [min_pages, max_pages] if the document is empty,
    or if the LLM call/response parsing fails — auto mode must never hard-fail
    the /research request.
    """
    fallback = (min_pages + max_pages) // 2

    doc = markdown_document.strip() if markdown_document else ""
    if not doc:
        logger.warning("[decide_num_pages] empty document, using fallback=%d", fallback)
        return fallback

    messages = [
        ChatMessage(role=Role.SYSTEM, content=_SYSTEM_PROMPT),
        ChatMessage(
            role=Role.USER,
            content=_USER_PROMPT_TEMPLATE.format(
                instruction=instruction or "(none provided)",
                document=_truncate_document(doc),
                min_pages=min_pages,
                max_pages=max_pages,
            ),
        ),
    ]

    try:
        response = await llm.run(messages=messages)
        text = response.choices[0].message.content or ""
        parsed = get_json_from_response(text)
        num_pages = int(parsed["num_pages"]) if isinstance(parsed, dict) else int(parsed[0])
    except Exception as e:
        logger.warning("[decide_num_pages] failed (%s), using fallback=%d", e, fallback)
        return fallback

    return max(min_pages, min(max_pages, num_pages))
