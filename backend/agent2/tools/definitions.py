"""
definitions.py — OpenAI tool schemas for agent2.

Six tools, in the order the agent should try them for a new outline request:
  1. match_outline_template    — vector search + LLM judge on pre-built templates
  2. search_outline_templates  — vector search only, returns top-N candidates (no LLM)
  3. load_template_outline     — load full outline for a specific template by scene_name
  4. build_outline_from_anchor — pure-Python subtree expand from agent-selected anchor node
  5. search_graph_tree         — FAISS search KB → build ancestor paths → return tree
  6. modify_outline            — patch current outline via natural-language instruction
"""

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "match_outline_template",
            "description": (
                "在预制大纲模板库中检索并由 LLM 判断最匹配的模板，决策是否可复用。"
                "status=pending_confirm 表示找到可用模板；status=not_found 表示无匹配，需改用 search_graph_tree → build_outline_from_anchor。"
                "用户提出新的分析需求时，优先调用此工具。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "用户的分析需求描述，原文传入",
                    }
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_outline_templates",
            "description": (
                "仅做向量检索，返回模板库中与需求最相似的 top-N 候选模板列表（不经 LLM 判断）。"
                "用于用户想直接浏览有哪些可用模板时调用，或在 match_outline_template 结果存疑时补充参考。"
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
                "按模板名称直接加载指定模板的完整大纲内容，跳过向量检索和 LLM 判断。"
                "当用户已从候选列表中看到某个模板名称，想查看其具体大纲结构时调用。"
                "scene_name 必须与 search_outline_templates 返回的候选名称完全一致。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "scene_name": {
                        "type": "string",
                        "description": "模板场景名称，与候选列表中的 scene_name 完全一致",
                    },
                },
                "required": ["scene_name"],
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
                "match_outline_template 返回 not_found 后必须先调用此工具，再从结果树中选锚节点调用 build_outline_from_anchor。"
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
                            "- add_node: {op, node_id, parent_id} — 从知识图谱新增节点到指定父节点下\n"
                            "- delete_node: {op, node_id} — 删除节点及其子树\n"
                            "- modify_node_name: {op, node_id, value} — 修改节点名称\n"
                            "- modify_node_description: {op, node_id, value} — 修改节点描述\n"
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
