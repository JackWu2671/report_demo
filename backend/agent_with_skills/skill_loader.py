"""
skill_loader.py — hermes-agent 风格的三级渐进式 skill 加载。

Level 0  skills_list()            → [{name, description, category}, ...]
Level 1  read_skill(name)         → 完整 SKILL.md 正文（SOP）
Level 2  read_skill(name, path)   → skills/<name>/<path> 指定文件内容
"""

import re
from pathlib import Path

_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\s*", re.DOTALL)


def _parse_frontmatter(text: str) -> dict:
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}
    result: dict = {}
    current_key: str | None = None
    hermes_block = False
    for line in m.group(1).splitlines():
        if line.strip() == "hermes:":
            hermes_block = True
            continue
        if hermes_block:
            if line.startswith("    ") and ":" in line:
                k, _, v = line.strip().partition(":")
                result[f"hermes.{k.strip()}"] = v.strip().strip('"')
                continue
            else:
                hermes_block = False
        if ":" in line and not line.startswith(" "):
            current_key, _, val = line.partition(":")
            result[current_key.strip()] = val.strip().strip('"')
    return result


def discover_skills(skills_dir: Path) -> list[dict]:
    """
    扫描 skills_dir 下所有 */SKILL.md（支持 category/skill-name/ 两级结构）。
    只读 frontmatter，返回 Level 0 元数据列表。
    """
    skills = []
    for skill_md in sorted(skills_dir.glob("**/SKILL.md")):
        text = skill_md.read_text(encoding="utf-8")
        meta = _parse_frontmatter(text)
        meta.setdefault("name", skill_md.parent.name)
        meta.setdefault("category", skill_md.parent.parent.name
                        if skill_md.parent.parent != skills_dir else "")
        meta["_path"] = skill_md.parent
        skills.append(meta)
    return skills


def read_skill(skill_path: Path, ref_path: str | None = None) -> str:
    """
    Level 1: ref_path=None  → 返回 SKILL.md 正文（frontmatter 之后的部分）
    Level 2: ref_path 指定  → 返回 skill 文件夹内的指定文件内容
    """
    if ref_path:
        target = skill_path / ref_path
        if not target.exists():
            return f"文件不存在: {ref_path}"
        return target.read_text(encoding="utf-8")

    text = (skill_path / "SKILL.md").read_text(encoding="utf-8")
    m = _FRONTMATTER_RE.match(text)
    return text[m.end():].strip() if m else text.strip()
