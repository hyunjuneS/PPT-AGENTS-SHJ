import asyncio
import json
import re
from itertools import cycle
from pathlib import Path
from typing import Any

import json_repair
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from pydantic import BaseModel, Field, PrivateAttr

from deeppresenter.utils.constants import RETRY_TIMES
from deeppresenter.utils.log import debug, logging_openai_exceptions


def get_json_from_response(response: str) -> dict | list:
    assert isinstance(response, str) and len(response) > 0, "response must be non-empty"
    response = response.strip()

    try:
        return json.loads(response)
    except Exception:
        pass

    # Find JSON by matching braces
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        starts = [i for i, c in enumerate(response) if c == start_char]
        ends = [i for i, c in enumerate(response) if c == end_char]
        for i in starts:
            for j in reversed(ends):
                if i >= j:
                    continue
                try:
                    obj = json.loads(response[i:j+1])
                    if isinstance(obj, (dict, list)):
                        return obj
                except Exception:
                    pass

    return json_repair.loads(response)


class LLM(BaseModel):
    model: str
    base_url: str | None = None
    api_key: str | None = None
    sampling_parameters: dict[str, Any] = Field(default_factory=dict)
    is_multimodal: bool | None = None

    _client: AsyncOpenAI | None = PrivateAttr(default=None)

    model_config = {"arbitrary_types_allowed": True}

    def model_post_init(self, _):
        self._client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )
        if self.is_multimodal is None:
            lower = self.model.lower()
            self.is_multimodal = any(w in lower for w in ("gpt", "claude", "gemini", "vl", "glm"))

    @property
    def model_name(self) -> str:
        return self.model.split("/")[-1]

    async def run(
        self,
        messages: list[Any],
        tools: list[dict] | None = None,
        response_format: type[BaseModel] | None = None,
        retry_times: int = RETRY_TIMES,
    ) -> ChatCompletion:
        from deeppresenter.utils.typings import ChatMessage

        # Convert ChatMessage objects to API dicts
        if messages and isinstance(messages[0], ChatMessage):
            api_messages = [m.to_api_dict() for m in messages]
        elif isinstance(messages, str):
            api_messages = [{"role": "user", "content": messages}]
        else:
            api_messages = messages

        errors = []
        for attempt in range(retry_times):
            try:
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": api_messages,
                    **self.sampling_parameters,
                }
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = "auto"
                if response_format is not None:
                    kwargs["response_format"] = response_format

                response = await self._client.chat.completions.create(**kwargs)

                assert response.choices and len(response.choices) > 0
                msg = response.choices[0].message
                assert msg.tool_calls or msg.content, "Empty response from model"

                return response

            except Exception as e:
                errors.append(str(e))
                logging_openai_exceptions(self.model, e)
                if attempt < retry_times - 1:
                    await asyncio.sleep(min(2 ** attempt, 30))

        raise ValueError(f"All {retry_times} retries failed:\n" + "\n".join(errors))


class DeepPresenterConfig(BaseModel):
    research_agent: LLM
    design_agent: LLM
    long_context_model: LLM
    context_window: int = 100_000
    max_context_folds: int = 3
    context_folding: bool = False
    multiagent_mode: bool = False
    offline_mode: bool = False

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)
