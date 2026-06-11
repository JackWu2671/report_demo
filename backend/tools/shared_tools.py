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
            "\n\n【修改 exec_sql 的硬性约束】"
            "\n- 只能在原始 SQL 的基础上做局部改动（枚举值替换、阈值调整、条件增减等）"
            "\n- 禁止添加原始 SQL 中不存在的字段名、表名或枚举值——系统无法查看表结构，新增内容大概率执行失败"
            "\n- 修改前必须先获取节点当前的 exec_sql，以原文为基础改写，不得凭空构造"
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
                    "description": (
                        "新值。exec_sql/name/description/condition 传字符串；condition_queries 传数组。"
                        "exec_sql 必须基于节点原有 SQL 改写，禁止引入原 SQL 中未出现的字段或枚举值。"
                    ),
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
            "\n\n【适用场景】用户/专家想自己组合报告结构时使用——典型是专家知识沉淀（把一段业务方法论组织成章节大纲），"
            "或用户明确要求按自定义结构搭建报告。本工具是整棵覆盖写入，会替换当前大纲。"
            "\n【不适用】在已有大纲上做局部改动：改节点属性用 edit_node，增删/保留节点用 modify_outline.py。"
            "\n\n【结构约束】"
            "\n1. outline 是节点对象数组，且恰好一个根节点(L1，报告总标题)"
            "\n2. 根节点的 name 即报告总标题，必须与用户/专家描述中已给出的标题完全一致——若已明确命名，原文照用，不得擅自摘要或改写"
            "\n3. 每个节点是对象 {id, name, description, children}，子节点放进 children 数组（无子节点可省略 children）"
            "\n4. L5 query 节点必须是叶子(无 children)，其 id 须引用 search_graph_tree 返回的知识库已有 id，禁止新建 query 节点"
            "\n5. 新建结构节点 id 必须按 new_L<层级>_<序号> 命名，显式编码层级：根用 new_L1_xxx，其下依次 new_L2_xxx / new_L3_xxx / new_L4_xxx（结构节点只能 L1~L4，绝不能 L5）"
            "\n6. 新建节点必须填写 description(50~100 字)"
            "\n7. 只写 id/name/description/children(及按需 condition/condition_queries)，不要写 level/exec_sql 等字段"
            "\n\n成功时工具会回显写入后的大纲；返回'写入失败'或未回显即为失败，须修正重试，不得告知用户已生成。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "outline": {
                    "type": "array",
                    "description": (
                        "【类型：array，禁止传字符串】"
                        "\n直接传 JSON 数组对象，绝对不能把数组序列化成字符串后传入。"
                        "\n错误示范（会解析失败）：\"outline\": \"[{\\\"id\\\": \\\"new_L1_root\\\", ...}]\""
                        "\n正确示范：\"outline\": [{\"id\": \"new_L1_root\", \"name\": \"报告标题\", \"description\": \"...\","
                        " \"children\": [{\"id\": \"new_L2_001\", \"name\": \"章节\", \"description\": \"...\","
                        " \"children\": [{\"id\": \"L5_001\", \"name\": \"指标名\"}]}]}]"
                        "\n顶层含一个 L1 根节点，每个节点含 id/name/description，子节点放 children 数组。"
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "id":          {"type": "string", "description": "节点 ID：新建结构节点用 new_L<层级>_<序号>，知识库叶子节点用原 id（如 L5_001）"},
                            "name":        {"type": "string", "description": "节点名称；根节点（L1）的 name 必须与用户/专家描述中已给出的标题完全一致，原文照用"},
                            "description": {"type": "string", "description": "节点描述，新建节点必填，50~100 字"},
                            "children":    {"type": "array",  "description": "子节点数组，L5 叶子节点不填此字段", "items": {"type": "object"}},
                            "condition":         {"type": "string"},
                            "condition_queries": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["id", "name"],
                    },
                },
            },
            "required": ["outline"],
        },
    },
}
