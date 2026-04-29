from typing import AsyncGenerator

from agent2.agent import Agent2
from skills.base import BaseSkill


class GenerateOutlineSkill(BaseSkill):
    name = "generate_outline"
    description = "根据用户描述生成报告大纲，支持多轮对话式修改。"

    def __init__(self) -> None:
        self._agent = Agent2()

    async def run(self, user_message: str) -> AsyncGenerator[dict, None]:
        async for event in self._agent.chat_stream(user_message):
            yield event

    def reset(self) -> None:
        self._agent.memory.reset()

    @property
    def outline(self) -> dict:
        """供上层编排器读取最终产物。"""
        return {
            "tree": self._agent.memory.outline_tree,
            "markdown": self._agent.memory.markdown,
        }
