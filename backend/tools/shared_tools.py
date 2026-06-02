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

SET_OUTLINE_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "set_outline",
        "description": (
            "一次性写入一份完整的报告大纲（结构由你自己组合）。"
            "参数 outline 是结构化 JSON 树，经 JSON 传递、完全不过 shell，不存在 YAML 缩进/heredoc 问题。"
            "\n\n【适用场景】用户/专家想自己组合报告结构时使用——典型是专家知识沉淀（把一段业务方法论组织成章节大纲）、"
            "或用户明确要求按自定义结构搭建报告。"
            "\n【不适用】在已有大纲上做局部改动：改节点属性用 edit_node，增删/保留节点用 modify_outline.py。"
            "本工具是整棵覆盖写入，会替换当前大纲。"
            "\n\n【结构约束】顶层恰好一个根节点(L1，报告总标题)；L5 query 节点必须是叶子，且其 id 必须引用"
            " search_graph_tree 返回的知识库已有 id，禁止新建 query 节点；新建的结构节点 id 以 new_ 开头并填写 description。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "outline": {
                    "type": "array",
                    "description": "大纲根节点列表（通常只含一个 L1 根）。每个节点见 items 结构。",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "description": "节点 id：新建结构节点用 new_xxx；L5 用知识库已有 id（如 L5_311）"},
                            "name": {"type": "string", "description": "节点名称"},
                            "description": {"type": "string", "description": "新建节点必填，50~100 字；L5 可省略"},
                            "condition": {"type": "string", "description": "可选。展示条件"},
                            "condition_queries": {"type": "array", "items": {"type": "string"}, "description": "可选"},
                            "children": {"type": "array", "description": "子节点列表，结构同本节点；L5 不可有 children"},
                        },
                        "required": ["id", "name"],
                    },
                },
            },
            "required": ["outline"],
        },
    },
}
