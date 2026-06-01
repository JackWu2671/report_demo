"""
shared_tools.py — AgentWithSkills 的元工具 schema 定义。
"""

READ_SKILL_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "read_skill",
        "description": (
            "加载指定 skill 的完整 SOP（Level 1），或其内部支持文件（Level 2）。"
            "决定使用某个 skill 前必须先加载其 SOP，已加载的 skill 无需重复加载。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "skill 名称，如 analyze-network"},
                "path": {
                    "type": "string",
                    "description": "可选。skill 文件夹内的支持文件路径（Level 2）",
                },
            },
            "required": ["name"],
        },
    },
}

BASH_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "执行 bash 命令，通常用于运行 skills/<name>/scripts/*.py 脚本工具。",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "要执行的 bash 命令"},
            },
            "required": ["command"],
        },
    },
}

EDIT_NODE_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "edit_node",
        "description": (
            "直接修改当前大纲中某个节点的属性值。"
            "参数经 JSON 传递、完全不过 shell，含反引号、<、>、% 的 SQL 或名称均可安全传入。"
            "修改 exec_sql / name / description / condition 等字段时优先用此工具，不要用 bash + modify_outline.py。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "string",
                    "description": "节点 ID，如 L5_071",
                },
                "field": {
                    "type": "string",
                    "enum": [
                        "exec_sql", "name", "description", "condition",
                        "summarySuggestion", "renderType", "colX", "colY",
                        "condition_queries",
                    ],
                    "description": "要修改的字段名",
                },
                "value": {
                    "description": "新值。exec_sql/name/description/condition 传字符串；condition_queries 传数组。",
                },
            },
            "required": ["node_id", "field", "value"],
        },
    },
}
