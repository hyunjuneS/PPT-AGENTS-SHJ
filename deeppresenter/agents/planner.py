from pathlib import Path

from deeppresenter.utils.log import info
from deeppresenter.utils.typings import ChatMessage, InputRequest, Role

from .agent import Agent


class Planner(Agent):

    async def loop(self, req: InputRequest):
        while True:
            agent_message = await self.action(
                prompt=req.deepresearch_prompt,
                attachments=req.attachments or None,
            )
            yield agent_message

            outcome = await self.execute(agent_message.tool_calls or [])

            if isinstance(outcome, str):
                outline_path = Path(outcome)
                if not outline_path.is_absolute():
                    outline_path = self.workspace / outline_path
                outcome = str(outline_path)
                info(f"Planner finished outline at {outcome}")

                while True:
                    feedback = yield outcome
                    if not feedback or not str(feedback).strip():
                        return

                    self.chat_history.append(
                        ChatMessage(role=Role.USER, content=str(feedback))
                    )

                    while True:
                        agent_message = await self.action(
                            prompt=req.deepresearch_prompt,
                            attachments=req.attachments or None,
                        )
                        yield agent_message

                        outcome = await self.execute(agent_message.tool_calls or [])

                        if isinstance(outcome, str):
                            outline_path = Path(outcome)
                            if not outline_path.is_absolute():
                                outline_path = self.workspace / outline_path
                            outcome = str(outline_path)
                            break

                        for item in outcome:
                            yield item

            else:
                for item in outcome:
                    yield item
