"""
base.py — BaseSkill abstract class.

All skills inherit from this. Enforces a stable interface so a future
orchestrator can discover and invoke any skill uniformly.

Convention (following CrewAI's BaseTool pattern):
  - name / description are class-level string attributes
  - run() is the async generator entry point
  - reset() clears session state (default: no-op)
"""

from abc import ABC, abstractmethod
from typing import AsyncGenerator


class BaseSkill(ABC):
    name: str = ""
    description: str = ""

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "name", ""):
            raise TypeError(f"{cls.__name__} must define a non-empty 'name' class attribute")
        if not getattr(cls, "description", ""):
            raise TypeError(f"{cls.__name__} must define a non-empty 'description' class attribute")

    @abstractmethod
    async def run(self, user_message: str) -> AsyncGenerator[dict, None]:
        """Drive one conversation turn, yielding typed SSE events."""
        ...

    def reset(self) -> None:
        """Clear session state. Override if the skill holds stateful resources."""
