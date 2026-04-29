"""
skill_loader.py — hermes-agent 风格的 skill 发现与内容加载。

discover_skills()   : 启动时扫描，只读 frontmatter（极少 IO）
load_skill_content(): 被调用时才读完整 SKILL.md body（渐进式加载）
"""

import re
from pathlib import Path

_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\s*", re.DOTALL)


def _parse_frontmatter(text: str) -> dict:
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
    扫描 skills_dir/*/SKILL.md，只解析 frontmatter，返回元数据列表。
    每个 dict 包含 name、description 及 _path（skill 文件夹路径）。
    """
    skills = []
    for skill_md in sorted(skills_dir.glob("*/SKILL.md")):
        text = skill_md.read_text(encoding="utf-8")
        meta = _parse_frontmatter(text)
        meta.setdefault("name", skill_md.parent.name)
        meta["_path"] = skill_md.parent
        skills.append(meta)
    return skills


def load_skill_content(skill_path: Path) -> str:
    """读取 SKILL.md 正文（frontmatter 之后的部分），注入 LLM 上下文。"""
    text = (skill_path / "SKILL.md").read_text(encoding="utf-8")
    m = _FRONTMATTER_RE.match(text)
    return text[m.end():].strip() if m else text.strip()
