from deeppresenter.agents.agent import Agent
from deeppresenter.utils.typings import InputRequest


class Design(Agent):

    async def loop(self, req: InputRequest, markdown_file: str, template_content: str = ""):
        (self.workspace / "slides").mkdir(exist_ok=True)

        while True:
            agent_message = await self.action(
                markdown_file=markdown_file,
                prompt=req.designagent_prompt,
                template_content=template_content,
            )
            yield agent_message

            outcome = await self.execute(agent_message.tool_calls or [])

            if isinstance(outcome, str):
                yield outcome
                break

            for item in outcome:
                yield item
