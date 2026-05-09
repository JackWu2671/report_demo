# Backend 详解

> 本文档面向第一次接触这个项目的开发者，从架构到每一行代码的设计意图都有说明。

---

## 目录

- [整体架构](#整体架构)
- [入口：api_server.py](#入口api_serverpy)
- [Agent 是什么，主循环怎么工作](#agent-是什么主循环怎么工作)
- [三个 Agent 的异同](#三个-agent-的异同)
- [工具系统：tools/ 和 handlers](#工具系统tools-和-handlers)
- [Memory：状态怎么管理](#memory状态怎么管理)
- [utils/：内部基础设施](#utils内部基础设施)
- [知识图谱检索全流程](#知识图谱检索全流程)
- [大纲修改：patcher.py](#大纲修改patcherpy)
- [AgentWithSkills 的 Skill 系统](#agentwithskills-的-skill-系统)
- [SSE 事件完整列表](#sse-事件完整列表)
- [环境变量](#环境变量)
- [本地调试](#本地调试)

---

## 整体架构

```
前端（React）
   │  POST /api/session  →  创建会话，绑定 Agent 实例
   │  POST /api/chat     →  发消息，接收 SSE 事件流
   ▼
api_server.py（FastAPI）
   │  session_id → Agent 实例（内存字典）
   ▼
agent.chat_stream(message)   ← async generator，一件事 yield 一个事件
   │
   ├─ 调 LLM（携带工具列表 + 对话历史）
   ├─ 执行工具（tools/ 实现 + utils/ 基础设施）
   └─ 更新 memory（大纲、元数据、对话历史）
```

---

## 入口：api_server.py

FastAPI 应用，提供三个接口：

| 接口 | 方法 | 功能 |
|------|------|------|
| `/api/session` | POST | 创建会话，`agent_id` 1/2/3 分别对应 Agent1/Agent2/AgentWithSkills |
| `/api/chat` | POST | 发消息，响应是 `text/event-stream`（SSE） |
| `/api/kb` | GET | 返回知识图谱节点 JSON（供前端知识库页面展示） |
| `/api/templates` | GET | 返回已保存的模板列表（供前端模板页面展示） |

**Session 生命周期**

```python
_sessions: dict[str, Agent1 | Agent2 | AgentWithSkills] = {}
```

Session 存在内存字典里，服务重启即清空。每个 session 绑定一个 Agent 实例，持有完整的对话历史和大纲状态。

**SSE 推流**

```python
async def _stream_agent(session_id, message):
    async for event in agent.chat_stream(message):
        yield f"data: {json.dumps(event)}\n\n"   # SSE 格式
    yield "data: [DONE]\n\n"
```

`chat_stream` 是 async generator，每发生一件事（工具开始/结束、大纲更新、文字回复）就 yield 一个 dict，这里原样序列化推给前端。

---

## Agent 是什么，主循环怎么工作

### Function Calling

大模型原生只能输出文字。OpenAI 协议扩展了一个 `tools` 参数，让我们可以给大模型一份"工具说明书"。大模型在合适时机不输出文字，而是输出：

```json
{
  "finish_reason": "tool_calls",
  "tool_calls": [{
    "id": "call_abc",
    "function": {
      "name": "search_graph_tree",
      "arguments": "{\"question\": \"fgOTN 高价值行业覆盖\"}"
    }
  }]
}
```

代码解析这个结构，执行对应工具，把结果放回对话历史，再调一次大模型，如此循环。大模型最终输出普通文字时，循环结束。这叫 **ReAct**（Reason + Act）。

### 主循环代码（agent2/agent.py）

```python
async def chat_stream(self, user_message: str):
    self.memory.add_message({"role": "user", "content": user_message})
    t0 = time.time()

    for _round in range(_MAX_TOOL_ROUNDS):     # 最多循环 N 轮，防止死循环
        response = await self._call_llm()
        choice = response.choices[0]
        msg = choice.message
        self.memory.add_message(msg.model_dump(exclude_none=True))

        if choice.finish_reason == "tool_calls" and msg.tool_calls:
            for tc in msg.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments)

                yield {"type": "step", "name": name, "status": "running",
                       "call_id": tc.id, "args": args}         # ① 通知前端"开始执行"

                result_dict, llm_str = await self._execute_tool(tc)

                if result_dict.get("outline_tree"):
                    yield {"type": "outline", ...}              # ② 大纲立刻推给前端

                yield {"type": "step", "name": name, "status": "done",
                       "call_id": tc.id, "result": ..., "detail": llm_str}  # ③ 通知"执行完毕"

                self.memory.add_message({
                    "role": "tool", "tool_call_id": tc.id, "content": llm_str
                })                                              # ④ 结果放入历史
            continue                                            # ⑤ 再调一次 LLM

        # finish_reason == "stop"
        if msg.content:
            yield {"type": "text", "chunk": msg.content}
        yield {"type": "done", "seconds": round(time.time() - t0, 1)}
        return
```

**为什么 outline 在工具执行完后立刻 yield，而不等大模型回复？**

大纲是工具计算出来的数据，已经确定，不需要大模型再"输出"一遍。立刻推送，前端瞬间渲染。如果等大模型回复，用户要多等一次 LLM 调用的延迟。

---

## 三个 Agent 的异同

| | Agent1 | Agent2 | AgentWithSkills |
|--|--------|--------|-----------------|
| **场景** | 专家沉淀知识 | 用户生成报告 | 同时支持两种场景 |
| **工具** | 5 个（专家流程） | 5 个（报告流程） | 全部 10 个 |
| **Memory** | Agent1Memory（含 extraction 字段） | AgentMemory（基类） | Agent1Memory（超集） |
| **特殊机制** | 无 | 无 | Skill 渐进式加载 |
| **入口** | `agent_id=1` | `agent_id=2` | `agent_id=3` |

**Agent1 的工具链**

```
search_graph_tree → set_outline_from_markdown → set_scene_metadata → [modify_outline] → save_outline_template
```

专家输入业务描述 → 检索图谱找锚点 → LLM 自行设计大纲结构（用 set_outline_from_markdown 写入）→ 填元数据 → 保存

**Agent2 的工具链**

```
search_outline_templates → load_template_outline          （有现成模板时）
search_graph_tree → build_outline_from_anchor → [modify_outline]  （从知识库实时构建时）
```

用户提需求 → 先找模板，找到直接用，找不到从知识库展开

**AgentWithSkills**

合并以上两者，通过 Skill SOP（自然语言工作流文档）告诉大模型什么情况用哪套流程，具体见 [Skill 系统](#agentwithskills-的-skill-系统)。

---

## 工具系统：tools/ 和 handlers

### 三层结构

```
tools/shared_tools.py       ← Schema 定义（JSON Schema，大模型看到的工具说明书）
agent*/tools/definitions.py ← 每个 agent 按需选取的工具子集
agent*/tools/handlers.py    ← 工具调度层（薄适配层，调实现、写 memory）
tools/*.py                  ← 工具具体实现（纯业务逻辑，无 LLM 调用）
```

**为什么要分这三层？**

- **Schema 和实现分离**：`shared_tools.py` 只管"大模型能看到什么"，`tools/*.py` 只管"代码怎么执行"，改其中一个不影响另一个
- **handler 是薄适配层**：负责把 agent 的 memory 对象传给工具实现，工具实现本身不感知 agent 或 memory
- **不同 agent 复用同一实现**：`modify_outline.py` 被 Agent1 和 Agent2 共用，但各自的 handler 里更新的是自己 memory 类型

### shared_tools.py 中的工具全集

```
业务工具（agent1 / agent2 各选 5 个）：
  SEARCH_GRAPH_TREE_TOOL          — 知识图谱向量检索
  MODIFY_OUTLINE_TOOL             — 大纲结构化修改
  SEARCH_OUTLINE_TEMPLATES_TOOL   — 模板向量检索
  LOAD_TEMPLATE_OUTLINE_TOOL      — 按 id 加载模板
  BUILD_OUTLINE_FROM_ANCHOR_TOOL  — 以锚节点展开子树
  SET_OUTLINE_FROM_MARKDOWN_TOOL  — 从 md_with_ids 渲染大纲（agent1 专用）
  SET_SCENE_METADATA_TOOL         — 填写模板元数据（agent1 专用）
  SAVE_OUTLINE_TEMPLATE_TOOL      — 保存模板文件（agent1 专用）

Skill 元工具（AgentWithSkills 专用）：
  SKILLS_LIST_TOOL                — 列出可用 skill
  READ_SKILL_TOOL                 — 读取 skill SOP 正文
```

### handler 的标准签名

```python
async def handle_xxx(args: dict, memory: AgentMemory) -> tuple[dict, str]:
    result = await xxx_tool(args["param"])           # 调工具实现
    if result["status"] == "success":
        memory.set_outline(...)                      # 写 memory
    llm_str = f"[xxx] status={result['status']}\n..." # 给 LLM 看的精简摘要
    return result, llm_str
```

**为什么返回两份数据？**

- `result`（完整）：agent 主循环用，判断要不要推 `outline` 事件
- `llm_str`（精简）：放进对话历史给 LLM 看。用精简版而不是完整 markdown，是因为大纲可能很长，每轮都放进历史会快速消耗 token

---

## Memory：状态怎么管理

### 基类 AgentMemory（memory/store.py）

```python
class AgentMemory:
    outline_tree: dict    # 大纲 JSON（代码逻辑用）
    markdown: str         # 大纲 Markdown（前端渲染用）
    md_with_ids: str      # 大纲带节点 ID（LLM 上下文用）
    kb_tree_text: str     # search_graph_tree 返回的树文本（暂存）
    _history: list[dict]  # 对话历史（不含大纲）
```

### Agent1Memory（agent1/memory.py）

在基类基础上增加：

```python
extraction: dict  # {scene_name, keywords, summary, usage_conditions}
                  # 由 set_scene_metadata 写入，save_outline_template 读取
```

### 大纲为什么不存进对话历史？

大纲会随着修改不断变化。如果存进历史，旧版本大纲会一直留着，LLM 会被旧版本干扰（"当前大纲有 3 节"vs"你说的大纲有 5 节"）。另外大纲可能很长，每轮都带着历史版本浪费 token。

### 大纲怎么注入 LLM 上下文？

每次调 LLM 前，`build_messages()` 把最新大纲动态拼入 system prompt：

```python
def build_messages(self, system_prompt: str) -> list[dict]:
    content = system_prompt
    if self.has_outline:
        content += f"\n\n## 当前大纲（可通过节点ID引用）\n\n{self.md_with_ids}"
    return [{"role": "system", "content": content}, *self._history]
```

**为什么拼入 system prompt，而不是追加一条新的 system 消息？**

Qwen 等模型要求 system 消息只能出现在对话的最开头，追加到末尾会报 400 错误。拼进第一条 system 消息内容里是通用做法。

### 大纲的三种格式

大纲在系统中同时存在三种格式，由 `utils/outline_utils.py` 从同一棵 `outline_tree` 派生：

| 字段 | 示例 | 谁用 |
|------|------|------|
| `outline_tree` | `{"id": "L1_001", "name": "fgOTN部署", "children": [...]}` | 代码逻辑（patcher 的输入输出） |
| `markdown` | `# fgOTN部署\n## 传送网分析` | 前端渲染给用户 |
| `md_with_ids` | `[L1 L1_001] fgOTN部署\n  [L2 L2_001] 传送网分析` | 注入 LLM，让大模型能精确引用节点 ID |

LLM 不能直接看普通 Markdown 修改大纲的原因：`modify_outline` 操作需要节点 ID（如 `delete_node L3_002`），普通 Markdown 没有 ID，LLM 只能靠名称猜，容易定位错。

---

## utils/：内部基础设施

这些模块不暴露给 LLM，只被 `tools/` 调用，属于内部实现细节。

| 文件 | 职责 |
|------|------|
| `loader.py` | 单次加载 FAISS 索引 + 知识图谱数据，进程内缓存 |
| `retriever.py` | embed_query → FAISS 检索 → 构建祖先路径 → 组装树 dict |
| `subtree.py` | 给定锚节点 ID，递归展开知识图谱子树 |
| `patcher.py` | 将 ops 列表（add/delete/rename…）应用到大纲树 |
| `template_selector.py` | 模板向量检索（FAISS） |
| `outline_utils.py` | 大纲三种格式互转：`to_clean_json`、`to_markdown`、`to_markdown_with_ids` |

---

## 知识图谱检索全流程

`search_graph_tree` 是整个系统最核心的数据处理步骤，完整流程：

```
用户问题（自然语言）
  │
  ▼ embed_query()
向量（float32 ndarray, shape=(1,1024)）
  │
  ▼ search_nodes()
FAISS 余弦相似度检索 → top-K 命中节点（带 score）
  │
  ▼ build_candidate_paths()
为每个命中节点补全祖先路径
  "传送网覆盖分析" → "政企业务 > fgOTN升级 > 传送网覆盖分析"
  │
  ▼ 展开子节点（递归）
为每个命中节点展开全部后代（避免相似度阈值把子节点过滤掉）
  │
  ▼ 组装树 dict
[
  {
    "id": "L2_001", "name": "fgOTN升级", "level": 2,
    "hit": false, "score": null,
    "children": [
      {"id": "L3_001", "name": "传送网覆盖分析", "level": 3,
       "hit": true, "score": 0.91, "children": [...]}
    ]
  }
]
```

**为什么要展开子节点？**

FAISS 只命中相似度超过阈值的节点，但用户可能需要子节点的内容。例如命中 L3"传送网覆盖分析"后，L4/L5 的具体分析维度也应该展示给 LLM，让它选合适的锚节点。

**`search_graph_tree` vs `build_outline_from_anchor` 分工**

- `search_graph_tree`：让 LLM 看到知识图谱里有什么，**由 LLM 选锚节点**（需要理解用户意图）
- `build_outline_from_anchor`：纯 Python，不调 LLM，给定锚节点 ID 递归展开全部子节点生成大纲（纯数据操作）

分工原因：锚节点选择是"理解意图"，LLM 擅长；子树展开是"数据操作"，Python 更快更稳定。

---

## 大纲修改：patcher.py

`modify_outline` 工具接收一个 `ops` 列表，由 `patcher.apply_patch()` 执行：

| op | 参数 | 含义 |
|----|------|------|
| `add_node` | `node_id, parent_id` | 从知识图谱取节点（递归展开子树）插入到指定父节点下 |
| `delete_node` | `node_id` | 删除节点及其全部子树 |
| `modify_node_name` | `node_id, value` | 修改节点名称 |
| `modify_node_description` | `node_id, value` | 修改节点描述 |
| `modify_node_condition` | `node_id, value` | 设置节点的展示条件（`value` 为空字符串表示删除条件） |
| `keep_only_node` | `node_id` | 保留该节点，同级其他节点全部删除 |

**ops 为什么由 Agent LLM 直接构造？**

Agent LLM 在 system prompt 里已经看到了当前大纲的 `md_with_ids`（含所有节点 ID），直接输出 ops 最自然，不需要 patcher 内部再调一次 LLM 重新推理。

**`keep_only_node` 的批量处理**

这个操作会删除同级兄弟，多个 `keep_only_node` 可能互相影响执行顺序。`apply_patch` 会先收集所有 `keep_only_node` 的目标 ID，批量处理，其余操作按顺序执行。

---

## AgentWithSkills 的 Skill 系统

`agent_with_skills/` 合并了 Agent1 + Agent2 的工具，通过 Skill SOP 动态决定用哪套流程。

### 三级渐进式加载

| 级别 | 内容 | Token 代价 |
|------|------|-----------|
| Level 0 | skill 名称 + 一句话描述（启动时注入 system prompt） | 极少 |
| Level 1 | 完整 SOP（步骤、工具调用顺序、注意事项）| 中等，按需加载 |
| Level 2 | SOP 内引用的支持文件 | 按需加载 |

大模型判断需要某个 skill 时，调用 `read_skill(name="generate-report")`，返回 SKILL.md 全文。之后大模型按 SOP 步骤调用业务工具。

### skills/ 目录结构

```
skills/
├── generate-report/
│   └── SKILL.md      # generate-report 的完整工作流 SOP
└── consolidate-expert/
    └── SKILL.md      # consolidate-expert 的完整工作流 SOP
```

每个 SKILL.md 包含：
- YAML front matter（name、description、category）
- 自然语言工作流步骤（工具调用顺序、判断分支、注意事项）

`skill_registry.py` 在启动时扫描 `skills/` 目录，读取所有 SKILL.md 的 front matter 构成 Level 0 列表；`skill_loader.py` 负责按需读取 SKILL.md 正文（Level 1）。

### 工具分发

```python
async def _execute_tool(self, tool_call):
    name = tool_call.function.name
    if name == "skills_list":
        return self._handle_skills_list()
    if name == "read_skill":
        return self._handle_read_skill(args)
    # 业务工具转发给合并后的 HANDLERS
    handler = _BUSINESS_HANDLERS.get(name)
    return await handler(args, self.memory)
```

`_BUSINESS_HANDLERS` 是 agent2 HANDLERS 和 agent1 HANDLERS 的合并，agent1 的同名 handler 覆盖 agent2（`modify_outline` 两者逻辑一致，取其一即可）。

---

## SSE 事件完整列表

所有 agent 的 `chat_stream` yield 的事件类型：

```python
# 工具开始执行
{"type": "step", "name": str, "status": "running", "call_id": str, "args": dict}

# 工具执行完毕
{"type": "step", "name": str, "status": "done",
 "call_id": str, "result": str,   # 单行摘要（前端步骤面板展示）
 "detail": str}                    # 完整工具结果（可展开查看）

# 大纲更新（工具执行完立刻推，不等 LLM）
{"type": "outline", "markdown": str, "md_with_ids": str, "outline_tree": dict}

# 场景元数据（agent1 / AgentWithSkills 的 set_scene_metadata 成功后）
{"type": "metadata",   "scene_name": str, "summary": str,
 "keywords": list, "usage_conditions": str}          # agent1
{"type": "extraction", "scene_name": str, "summary": str,
 "keywords": list}                                   # AgentWithSkills

# 模板保存成功
{"type": "saved", "scene_name": str, "path": str}

# LLM 文字回复
{"type": "text", "chunk": str}

# 本轮结束
{"type": "done", "seconds": float}

# 出错
{"type": "error", "message": str}
```

---

## 环境变量

`.env` 文件（放在 `backend/` 目录下）：

| 变量 | 说明 | 示例 |
|------|------|------|
| `LLM_BASE_URL` | LLM API 地址（OpenAI 兼容） | `http://localhost:8080/v1` |
| `LLM_API_KEY` | LLM API Key | `sk-xxx` |
| `LLM_MODEL` | 模型名称 | `Qwen3-32B` |
| `LLM_TEMPERATURE` | 采样温度 | `0.7` |
| `EMBEDDING_BASE_URL` | Embedding 服务地址 | `http://localhost:8001/v1` |
| `EMBEDDING_DIM` | 向量维度 | `1024` |
| `FAISS_SCORE_THRESHOLD` | FAISS 检索相似度阈值 | `0.3` |

---

## 本地调试

### 启动服务

```bash
cd backend
uvicorn api_server:app --reload --port 8000
```

### 命令行交互测试（推荐，无需启动前端）

```bash
cd backend
python agent2/agent_test.py               # 测试 Agent2
python agent1/agent_test.py               # 测试 Agent1
python agent_with_skills/agent_test.py    # 测试 AgentWithSkills
```

内置命令：

| 命令 | 功能 |
|------|------|
| `/state` | 打印当前大纲 JSON |
| `/md` | 打印当前 Markdown 大纲（用户视图） |
| `/ids` | 打印当前带 ID 大纲（LLM 视图） |
| `/reset` | 重置会话（清空历史和大纲） |

### 离线构建向量索引

```bash
cd backend
python scripts/build_index.py
```

在 `expert_knowledge/` 中的 `node.json` 变更后需要重新构建。

### 单轮工具调用调试

```bash
cd backend
python tests/tool_test.py "帮我分析一下fgOTN的部署情况"
```

直接打印大模型原始 function calling 输出（JSON 结构），不走完整的 Agent 循环，用于排查工具定义或 prompt 问题。
