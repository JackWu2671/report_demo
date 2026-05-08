"""
definitions.py — agent1 的 OpenAI 工具 schema 定义。

共四个工具，按专家知识沉淀流程排列：
  1. search_graph_tree         — FAISS 检索知识库，返回带祖先路径的树状结构
  2. set_outline_from_markdown — LLM 构造 md_with_ids 文本后调此工具渲染为大纲
  3. modify_outline            — 对当前大纲执行结构化 patch 操作
  4. save_outline_template     — 将当前大纲保存为可复用模板
"""

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_graph_tree",
            "description": (
                "从知识图谱中检索与问题相关的节点，返回带祖先路径的树状结构（含节点 id、名称、描述）。"
                "专家提供场景描述后首先调用，获取可用的节点 id，再组合构造大纲。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "专家的业务场景描述，原文传入",
                    },
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_outline_from_markdown",
            "description": (
                "将 LLM 构造的 md_with_ids 格式大纲文本解析为结构化大纲，渲染到前端，并记录场景元数据。"
                "L2/L3/L4 层级由 LLM 按专家意图自由设计；L5 必须引用 search_graph_tree 返回的知识库节点 id。"
                "md_with_ids 格式：每行 {缩进}[L{层级} {节点id}] {节点名称}，缩进每层两个空格。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "scene_name": {
                        "type": "string",
                        "description": "场景名称，中文，不超过 10 字，如「传送网络覆盖分析」",
                    },
                    "summary": {
                        "type": "string",
                        "description": "一句话场景摘要，不超过 50 字，概括本次分析的核心目标",
                    },
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "3～8 个核心领域关键词，名词短语为主，代表分析维度、评估指标或技术名词",
                    },
                    "usage_conditions": {
                        "type": "string",
                        "description": "适用条件，说明在什么业务场景下适合使用这份大纲，以及有哪些前提要求，不超过 80 字",
                    },
                    "md_with_ids": {
                        "type": "string",
                        "description": (
                            "大纲文本，每行格式：{缩进}[L{层级} {id}] {名称}（新建节点名后加全角冒号和描述）\n"
                            "示例：\n"
                            "[L3 new_001] 传送网络覆盖分析：分析OTN站点对企业的覆盖情况\n"
                            "  [L4 new_002] 企业分布分析：从行业、行政区等维度统计企业分布\n"
                            "    [L5 L5_001] 企业行业分布\n"
                            "    [L5 L5_002] 企业行政区分布"
                        ),
                    },
                },
                "required": ["scene_name", "summary", "keywords", "usage_conditions", "md_with_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "modify_outline",
            "description": (
                "对当前报告大纲执行修改，直接传入结构化操作列表。"
                "仅当已存在大纲时可用。每次只传一个 op。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ops": {
                        "type": "array",
                        "description": (
                            "操作列表，每条操作包含 op 字段和对应参数。每次只传一个 op。\n"
                            "支持的操作：\n"
                            "- add_node: {op, node_id, parent_id} — 新增知识库已有节点（node_id 必须来自 search_graph_tree 返回结果，不可新建）；顶层章节 parent_id 传 \"\"\n"
                            "- delete_node: {op, node_id} — 删除节点及其子树\n"
                            "- modify_node_name: {op, node_id, value} — 修改节点名称\n"
                            "- modify_node_description: {op, node_id, value} — 修改节点描述\n"
                            "- keep_only_node: {op, node_id} — 保留该节点，删除同级其他节点"
                        ),
                        "items": {"type": "object"},
                    }
                },
                "required": ["ops"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_outline_template",
            "description": (
                "将当前大纲保存为可复用模板。"
                "仅在专家明确确认（如说'保存'、'好的就这样'）时调用，不得主动触发。"
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]
