"""
generate-outline skill — Python 实现层。

包装 agent2/agent.py，对外暴露统一的 Skill 接口。
通过 skill_loader.load_skill_class() 动态加载，不走 Python import 机制，
所以文件夹名可以保持 hermes-agent 的连字符约定。
"""

import os
import sys

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from agent2.agent import Agent2


class Skill:
    name = "generate-outline"
    description = "根据用户描述生成报告大纲，支持多轮对话式修改。"

    def __init__(self) -> None:
        self._agent: Agent2 | None = None  # 懒初始化，第一次 run() 时才创建

    async def run(self, user_message: str):
        if self._agent is None:
            self._agent = Agent2()
        async for event in self._agent.chat_stream(user_message):
            yield event

    def reset(self) -> None:
        if self._agent is not None:
            self._agent.memory.reset()

    @property
    def outline(self) -> dict:
        """供编排层读取最终产物。"""
        if self._agent is None:
            return {"tree": {}, "markdown": ""}
        return {
            "tree": self._agent.memory.outline_tree,
            "markdown": self._agent.memory.markdown,
        }
