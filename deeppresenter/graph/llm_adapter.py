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
    return ChatOpenAI(
        model=llm.model,
        base_url=llm.base_url,
        api_key=llm.api_key,
        timeout=_DEFAULT_TIMEOUT,
        **llm.sampling_parameters,
    )
