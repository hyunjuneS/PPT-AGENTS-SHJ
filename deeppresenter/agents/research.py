from pathlib import Path

from deeppresenter.utils.typings import InputRequest

from .agent import Agent


class Research(Agent):

    async def loop(self, req: InputRequest, outline_path: Path | None = None):
        while True:
            agent_message = await self.action(
                prompt=req.deepresearch_prompt,
                outline_path=str(outline_path) if outline_path else None,
            )
            yield agent_message

            outcome = await self.execute(agent_message.tool_calls or [])

            if isinstance(outcome, str):
                yield outcome
                break

            for item in outcome:
                yield item
