"""Adapts deeppresenter.utils.config.LLM (the existing role->model config, still
loaded the same way from DeepPresenterConfig) into a LangChain ChatOpenAI instance.

Note: this app always talks to its configured model through an OpenAI-compatible
`base_url` (see .env.example's OPENAI_BASE_URL / DESIGN_MODEL_NAME) even when that
model is Claude — so ChatOpenAI is the correct wrapper here, not ChatAnthropic.
"""

from langchain_openai import ChatOpenAI

from deeppresenter.utils.config import LLM


def to_chat_openai(llm: LLM) -> ChatOpenAI:
    return ChatOpenAI(
        model=llm.model,
        base_url=llm.base_url,
        api_key=llm.api_key,
        **llm.sampling_parameters,
    )
