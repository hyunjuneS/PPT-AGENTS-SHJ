"""Adapts deeppresenter.utils.config.LLM (the existing role->model config, still
loaded the same way from DeepPresenterConfig) into a LangChain ChatOpenAI instance.

Note: this app always talks to its configured model through an OpenAI-compatible
`base_url` (see .env.example's OPENAI_BASE_URL / VLM_MODEL_URL) even when that
model is Claude — so ChatOpenAI is the correct wrapper here, not ChatAnthropic.
"""

from openai import Timeout
from langchain_openai import ChatOpenAI

from deeppresenter.utils.config import LLM

# Matches openai.AsyncOpenAI's own default when no timeout is passed explicitly
# (confirmed via inspection: AsyncOpenAI(api_key=...).timeout). ChatOpenAI does
# NOT resolve to this same default on its own — when constructed without an
# explicit `timeout`, it omits the `x-stainless-read-timeout` header entirely
# (confirmed by capturing the actual outgoing request), instead of sending the
# same "600s" declaration the old engine's raw AsyncOpenAI client always sent.
# Some gateways/proxies use that header to size their own upstream wait time,
# so its absence can make a gateway cut the connection early on slow responses
# that would have succeeded under the old engine — reproducing it explicitly
# here keeps the new engine's declared patience identical to the old one.
_DEFAULT_TIMEOUT = Timeout(connect=5.0, read=600, write=600, pool=600)


def to_chat_openai(llm: LLM) -> ChatOpenAI:
    # Default max_tokens: without an explicit cap, some gateways/backends fall back to
    # a much smaller default output limit of their own — enough for short replies, but
    # not enough to finish generating a whole slide's HTML inside a single write_file
    # tool-call argument. When that default is too small, the response gets cut off
    # mid-argument (finish_reason "length"), producing truncated/invalid JSON that
    # write_file's tool call fails to parse — confirmed by reproducing it with a short
    # write_file (succeeds) vs. a longer one (fails) on the same model. 16384 is well
    # under GLM-5.3-flash's real ~128K output ceiling, so this is just raising the
    # floor, not asking the model to generate more than it would anyway.
    # llm.sampling_parameters is spread last so a caller-supplied max_tokens (e.g. via
    # the API's additional_request param) still overrides this default.
    params = {"max_tokens": 16384, **llm.sampling_parameters}
    return ChatOpenAI(
        model=llm.model,
        base_url=llm.base_url,
        api_key=llm.api_key,
        timeout=_DEFAULT_TIMEOUT,
        # openai SDK's own client-level retry (default 2, logged as "Retrying
        # request to ... in Ns" from openai._base_client) is kept as-is — left
        # unset here so it stays at its default rather than being disabled. It's
        # a second retry layer nested inside every attempt engine.py's own
        # _invoke_with_retry sees, so .history/{agent}-llm-calls.json records one
        # "failed" entry per outer attempt even when the SDK silently retried a
        # few times underneath it first.
        **params,
    )
