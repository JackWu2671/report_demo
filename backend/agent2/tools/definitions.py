"""
definitions.py — agent2 的 OpenAI 工具 schema 定义。

共五个工具，按新建大纲时的推荐调用顺序排列：
  1. search_outline_templates  — 向量检索，返回 top-N 候选模板列表（纯检索，不调 LLM）
  2. load_template_outline     — 按模板名称直接加载完整大纲
  3. build_outline_from_anchor — 从 agent 选定的锚节点展开知识图谱子树
  4. search_graph_tree         — FAISS 检索知识库，构建带祖先路径的树状结构
  5. modify_outline            — 对当前大纲执行结构化 patch 操作
"""

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_outline_templates",
            "description": (
                "向量检索模板库，返回与需求最相似的 top-N 候选模板列表（含 scene_name、summary、score）。"
                "用户提出新的分析需求时优先调用。根据返回的候选列表自行判断是否有匹配的模板："
                "有匹配 → 调用 load_template_outline 加载；无匹配 → 调用 search_graph_tree 从知识库生成。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "用户的分析需求描述，原文传入",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回候选数量，默认 5",
                    },
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_template_outline",
            "description": (
                "按模板 id 直接加载指定模板的完整大纲内容。"
                "在 search_outline_templates 返回候选后，判断有匹配时调用此工具加载大纲，再询问用户是否使用。"
                "template_id 必须取自 search_outline_templates 返回的候选列表中的 id 字段。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "template_id": {
                        "type": "string",
                        "description": "模板唯一 id，取自 search_outline_templates 返回的候选列表中的 id 字段",
                    },
                },
                "required": ["template_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "build_outline_from_anchor",
            "description": (
                "以指定节点为根，从知识图谱展开子树，生成初始报告大纲。"
                "必须在 search_graph_tree 成功后，从返回的树中选出最相关节点的 id，再调用此工具。"
                "anchor_id 取自 search_graph_tree 返回的树节点 id 字段，选择与用户需求最直接相关的节点。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "anchor_id": {
                        "type": "string",
                        "description": "锚节点 id，从 search_graph_tree 返回的树中选取，如 'L4_001'",
                    }
                },
                "required": ["anchor_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_graph_tree",
            "description": (
                "从知识图谱中检索与问题相关的节点，返回带祖先路径的树状结构（含节点 id、描述、FAISS 命中分数）。"
                "search_outline_templates 无合适候选时调用此工具，再从结果树中选锚节点调用 build_outline_from_anchor。"
                "status=not_found 表示知识库无相关内容，应告知用户系统暂不支持该场景。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "用户的分析需求描述，原文传入",
                    },
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "modify_outline",
            "description": (
                "对当前报告大纲执行修改，直接传入结构化操作列表。"
                "仅当已存在大纲时可用。ops 由你根据用户指令和当前大纲（system prompt 中）直接构造。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ops": {
                        "type": "array",
                        "description": (
                            "操作列表，每条操作包含 op 字段和对应参数。\n"
                            "支持的操作：\n"
                            "- add_node: {op, node_id, parent_id} — 从知识图谱新增节点到指定父节点下；若要新增与现有一级章节平行的顶层章节，parent_id 传空字符串 \"\"\n"
                            "- delete_node: {op, node_id} — 删除节点及其子树\n"
                            "- modify_node_name: {op, node_id, value} — 修改节点名称\n"
                            "- modify_node_description: {op, node_id, value} — 修改节点描述\n"
                            "- modify_node_condition: {op, node_id, value} — 设置或修改节点展示条件；value 格式必须为「当……时，本节才展示」；value 传空字符串表示删除条件\n"
                            "- keep_only_node: {op, node_id} — 保留该节点，删除同级其他节点（每个保留节点单独一条）"
                        ),
                        "items": {"type": "object"},
                    }
                },
                "required": ["ops"],
            },
        },
    },
]
