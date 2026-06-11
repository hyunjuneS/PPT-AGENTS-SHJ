import base64
import logging
from dataclasses import dataclass, field

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

MAX_CONTEXT_SIZE = 32768


def tenacity_decorator(func):
    return retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )(func)


@dataclass
class AsyncLLM:
    """Async wrapper for OpenAI-compatible LLM APIs."""

    model: str
    base_url: str | None = None
    api_key: str | None = None
    timeout: int = 360

    def __post_init__(self):
        self._client: AsyncOpenAI | None = None

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout,
            )
        return self._client

    @tenacity_decorator
    async def __call__(
        self,
        content: str,
        images: str | list[str] | None = None,
        system_message: str | None = None,
        history: list | None = None,
        return_json: bool = False,
        return_message: bool = False,
        **client_kwargs,
    ) -> str | dict | tuple:
        if history is None:
            history = []

        system, message = self.format_message(content, images, system_message)

        try:
            completion = await self.client.chat.completions.create(
                model=self.model,
                messages=system + history + message,
                **client_kwargs,
            )
        except Exception as e:
            logger.error("LLM call failed (model=%s): %s", self.model, e)
            raise

        response = completion.choices[0].message.content
        message.append({"role": "assistant", "content": response})

        response = response.strip()
        if return_message:
            return response, message
        return response

    def format_message(
        self,
        content: str,
        images: str | list[str] | None = None,
        system_message: str | None = None,
    ) -> tuple[list, list]:
        if isinstance(images, str):
            images = [images]

        if len(content) > MAX_CONTEXT_SIZE:
            logger.warning("Input may be too long: %d chars", len(content))

        if system_message is None:
            system_message = "You are a helpful assistant."

        system = [{"role": "system", "content": [{"type": "text", "text": system_message}]}]
        message = [{"role": "user", "content": [{"type": "text", "text": content}]}]

        if images:
            for img_path in images:
                try:
                    with open(img_path, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode("utf-8")
                        message[0]["content"].append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                            }
                        )
                except Exception as e:
                    logger.error("Failed to load image %s: %s", img_path, e)

        return system, message

    def __repr__(self) -> str:
        base = f"AsyncLLM(model={self.model}"
        if self.base_url:
            base += f", base_url={self.base_url}"
        return base + ")"
