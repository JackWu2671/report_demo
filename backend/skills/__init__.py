from skills.base import BaseSkill
from skills.generate_outline import GenerateOutlineSkill

# Registry: skill name → skill class.
# A future orchestrator imports this dict to discover all available skills.
SKILL_REGISTRY: dict[str, type[BaseSkill]] = {
    GenerateOutlineSkill.name: GenerateOutlineSkill,
}

__all__ = ["BaseSkill", "GenerateOutlineSkill", "SKILL_REGISTRY"]
