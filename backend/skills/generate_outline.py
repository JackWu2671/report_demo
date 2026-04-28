"""
Skill: 生成大纲

根据用户描述生成报告大纲，支持多轮对话式修改。
内部封装 Agent2，对外暴露统一的 skill 接口。

上层编排器（未来）通过 skill name 找到此 skill，调用 run() 驱动对话，
通过 outline 属性读取最终产物。
"""

from typing import AsyncGenerator

from agent2.agent import Agent2

SKILL_NAME = "generate_outline"
SKILL_DESCRIPTION = "根据用户描述生成报告大纲，支持多轮对话式修改。"


class GenerateOutlineSkill:
    """
    生成大纲 skill。

    每个用户会话创建一个实例，会话结束后调用 reset() 或丢弃实例。
    run() 产出的事件格式与 Agent2.chat_stream() 完全一致，前端无需感知。
    """

    def __init__(self) -> None:
        self._agent = Agent2()

    async def run(self, user_message: str) -> AsyncGenerator[dict, None]:
        """驱动一轮对话，透传 Agent2 的所有事件。"""
        async for event in self._agent.chat_stream(user_message):
            yield event

    def reset(self) -> None:
        """清空对话历史和大纲状态，复用同一实例开启新会话。"""
        self._agent.memory.reset()

    @property
    def outline(self) -> dict:
        """返回当前大纲，供上层编排器读取最终产物。"""
        return {
            "tree": self._agent.memory.outline_tree,
            "markdown": self._agent.memory.markdown,
        }
