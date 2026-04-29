"""
skill_loader.py — hermes-agent 风格的 skill 发现与加载。

启动时只扫描 SKILL.md frontmatter（极少 token/IO）。
Python 实现（skill.py）在 skill 第一次被调用时才动态加载。
"""

import importlib.util
import re
import sys
from pathlib import Path

_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\s*", re.DOTALL)


def _parse_frontmatter(text: str) -> dict:
    """Extract YAML-like key: value pairs from --- frontmatter block."""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}
    result = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            result[key.strip()] = val.strip().strip('"')
    return result


def discover_skills(skills_dir: Path) -> list[dict]:
    """
    Scan skills_dir for */SKILL.md files, return list of metadata dicts.
    Each dict has at minimum: name, description, _path (Path to skill folder).
    Mirrors hermes-agent agent/skill_utils.py::iter_skill_index_files().
    """
    skills = []
    for skill_md in sorted(skills_dir.glob("*/SKILL.md")):
        text = skill_md.read_text(encoding="utf-8")
        meta = _parse_frontmatter(text)
        meta.setdefault("name", skill_md.parent.name)  # fallback to folder name
        meta["_path"] = skill_md.parent
        skills.append(meta)
    return skills


def load_skill_class(skill_path: Path):
    """
    Dynamically load skill.py from a skill folder, return the Skill class.
    Uses importlib because the folder name may contain hyphens.
    """
    skill_py = skill_path / "skill.py"
    module_name = f"_skill_{skill_path.name.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, skill_py)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod.Skill
