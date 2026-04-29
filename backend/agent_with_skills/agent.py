"""
agent.py — skill 编排层。

启动时从 skills/ 扫描元数据，首次调用某个 skill 时才动态加载其 Python 实现。
当前只有一个 skill（generate-outline），直接路由；
后续增加 skill 时，这里可以接入 LLM 做意图路由。
"""

import logging
from pathlib import Path
from typing import AsyncGenerator

from agent_with_skills.skill_loader import discover_skills, load_skill_class

logger = logging.getLogger(__name__)

_SKILLS_DIR = Path(__file__).parent / "skills"


class AgentWithSkills:
    def __init__(self) -> None:
        # 启动时只加载元数据，不初始化任何 Python 实现
        self._skill_meta: list[dict] = discover_skills(_SKILLS_DIR)
        self._instances: dict[str, object] = {}
        logger.info(
            "[AgentWithSkills] discovered skills: %s",
            [m["name"] for m in self._skill_meta],
        )

    def _get_skill(self, name: str):
        """懒加载：第一次被调用时才 import + 实例化 skill.py。"""
        if name not in self._instances:
            meta = next((m for m in self._skill_meta if m["name"] == name), None)
            if meta is None:
                raise ValueError(f"skill not found: {name!r}")
            SkillClass = load_skill_class(meta["_path"])
            self._instances[name] = SkillClass()
            logger.info("[AgentWithSkills] loaded skill: %s", name)
        return self._instances[name]

    async def chat_stream(self, user_message: str) -> AsyncGenerator[dict, None]:
        """
        当前单 skill 直接路由到 generate-outline。
        多 skill 时在此处加 LLM 意图判断，选择对应 skill name。
        """
        skill = self._get_skill("generate-outline")
        async for event in skill.run(user_message):
            yield event

    def reset(self, skill_name: str = "generate-outline") -> None:
        """清空指定 skill 的会话状态。"""
        if skill_name in self._instances:
            self._instances[skill_name].reset()
