# Backend 详解

> 面向第一次接触本项目的开发者，说明当前**脚本驱动单 Agent** 架构的设计意图。

---

## 目录

- [整体架构](#整体架构)
- [入口：api_server.py](#入口api_serverpy)
- [Agent 主循环](#agent-主循环)
- [四个工具：read_skill / bash / edit_node / set_outline](#四个工具read_skill--bash--edit_node--set_outline)
- [Session 文件桥：内存 ↔ 脚本](#session-文件桥内存--脚本)
- [Skill 系统：渐进式加载](#skill-系统渐进式加载)
- [Memory：状态怎么管理](#memory状态怎么管理)
- [大纲修改：patcher.py](#大纲修改patcherpy)
- [报告生成与 mock 取数](#报告生成与-mock-取数)
- [SSE 事件完整列表](#sse-事件完整列表)
- [目录结构](#目录结构)
- [环境变量](#环境变量)
- [本地调试](#本地调试)

---

## 整体架构

系统只有**一个 Agent**：`AgentWithSkills`。它本身几乎**不含业务逻辑**，绝大多数业务能力以 Python 脚本形式存放在 `skills/<name>/scripts/`，LLM 通过 SKILL.md（自然语言 SOP）了解脚本的 CLI 接口，再用 `bash` 调用——LLM 不需要为每个能力维护 JSON tool schema。

唯一的例外是 `edit_node`、`set_outline` 两个原生工具：它们确实带业务语义（认识大纲节点、`exec_sql` 等字段），之所以不做成脚本，是因为它们的参数常含反引号/`<`/`>` 的 SQL 或整棵结构化大纲，走 bash 会被 shell 破坏（详见[四个工具](#四个工具read_skill--bash--edit_node--set_outline)）。除这两个窄口子外，业务逻辑仍全部在脚本里。

```
客户端（HTTP / SSE）
   │  POST /api/session  →  创建会话，绑定一个 AgentWithSkills 实例
   │  POST /api/chat     →  发消息，接收 SSE 事件流（聊天回合）
   │  POST /api/report   →  渲染报告，接收 SSE 事件流（独立流）
   ▼
api_server.py（FastAPI）
   │  session_id → AgentWithSkills 实例（内存字典）
   ▼
agent.chat_stream(message)   ← async generator，一件事 yield 一个事件
   │
   ├─ 调 LLM（只带 read_skill / bash / edit_node / set_outline 四个工具）
   ├─ read_skill  → 读 SKILL.md SOP
   ├─ bash        → 跑 skills/*/scripts/*.py（业务逻辑都在这里）
   ├─ edit_node   → 直接改大纲节点属性（参数走 JSON，不过 shell）
   └─ set_outline → 整棵覆盖写入大纲（参数 JSON 数组，不过 shell）
```

**为什么这么设计？** 业务逻辑放进脚本，LLM 只需理解 SKILL.md 文档化的 CLI，不必为每个能力维护 JSON tool schema；新增能力 = 加一个脚本 + 在 SKILL.md 写一行，不动 agent 代码。

---

## 入口：api_server.py

FastAPI 应用，主要接口：

| 接口 | 方法 | 功能 |
|------|------|------|
| `/api/session` | POST | 创建会话，返回 `session_id`（`agent_id` 字段保留兼容，实际只有 AgentWithSkills） |
| `/api/chat` | POST | 发消息，响应是 `text/event-stream`（聊天回合 SSE） |
| `/api/report` | POST | 渲染报告，响应是 SSE（与 `/api/chat` 相互独立的流） |
| `/api/session/{id}/messages` | GET | 拉取会话历史消息 |
| `/api/kb` | GET | 返回知识图谱节点 JSON |
| `/api/templates` | GET | 返回已保存模板列表 |

**Session 生命周期**

```python
_sessions: dict[str, AgentWithSkills] = {}
```

存在内存字典，服务重启即清空。每个 session 绑定一个 Agent 实例，持有对话历史和大纲状态。

**SSE 推流**

```python
async for event in agent.chat_stream(message):
    yield f"data: {json.dumps(event)}\n\n"
yield "data: [DONE]\n\n"
```

`chat_stream` 是 async generator，每发生一件事（工具开始/结束、大纲更新、文字回复）就 yield 一个 dict，原样序列化推给客户端。

> `/api/chat`（聊天回合）和 `/api/report`（报告渲染）是**两条独立的 SSE 流**。触发报告后，agent 立刻结束聊天回合（推送固定回复并 `done`），报告在 `/api/report` 流里单独渲染——避免两个流抢占后端导致聊天流迟迟不关闭、调用方读不到流结束标志。

---

## Agent 主循环

LLM 通过 OpenAI 协议的 `tools` 参数感知四个工具。它在合适时机输出 `tool_calls`，代码执行后把结果放回历史，再调一次 LLM，循环往复（ReAct）。

```python
async def chat_stream(self, user_message: str):
    self.memory.add_message({"role": "user", "content": user_message})
    for _ in range(_MAX_ROUNDS):
        response = await self._call_llm()          # 带 read_skill/bash/edit_node/set_outline
        msg = response.choices[0].message
        self.memory.add_message(msg.model_dump(exclude_none=True))

        if msg.tool_calls:
            report_triggered = False
            for tc in msg.tool_calls:
                yield {"type": "step", "status": "running", ...}    # ① 开始
                result_dict, llm_str = await self._execute_tool(name, args)
                for event in result_dict.get("_events", []):        # ② 状态变化事件
                    if event["type"] == "start_report":
                        report_triggered = True
                    yield event
                yield {"type": "step", "status": "done", ...}       # ③ 结束
                self.memory.add_message({"role": "tool", ...})      # ④ 结果入历史

            if report_triggered:                    # 触发报告 → 立刻结束本回合
                yield {"type": "text", "chunk": "好的，开始生成报告。"}
                yield {"type": "done", ...}
                return
            continue                                # ⑤ 再调一次 LLM

        if msg.content:
            yield {"type": "text", "chunk": msg.content}
        yield {"type": "done", ...}
        return
```

---

## 四个工具：read_skill / bash / edit_node / set_outline

定义在 `tools/shared_tools.py`（`READ_SKILL_TOOL` / `BASH_TOOL` / `EDIT_NODE_TOOL` / `SET_OUTLINE_TOOL`），在 `agent.py` 的 `_execute_tool` 中分发。

| 工具 | 作用 | 关键点 |
|------|------|--------|
| `read_skill(name, path?)` | 加载 SKILL.md SOP（Level 1）或其支持文件（Level 2） | 同一 skill 一个会话只加载一次 |
| `bash(command)` | 执行命令，通常是 `skills/*/scripts/*.py` | 调用前后做 [session 文件同步](#session-文件桥内存--脚本)；`$SKILLS_DIR` 等变量由 harness 预先展开后再交给 shell |
| `edit_node(node_id, field, value)` | 直接修改大纲节点属性 | **参数走 JSON、不过 shell**，含反引号/`<`/`>`/`%` 的 SQL 也安全 |
| `set_outline(outline)` | 一次性整棵覆盖写入大纲（专家自组结构等场景） | **参数是 JSON 节点数组、不过 shell**，结构靠括号承载、不依赖换行缩进 |

**`edit_node` / `set_outline` 为什么单独做成工具？** 改 `exec_sql`、`name` 这类字段，值常含反引号、`<`、`>`，走 bash 会被 cmd.exe 当重定向/命令替换破坏（静默失败）；整棵大纲若走 YAML 字符串，又会因 LLM 把换行压成一行导致解析失败。两者的共同解法：工具参数是 LLM 产出的 JSON，经 `json.loads` 直接入 Python，**完全不过 shell、不依赖空白格式**。

`edit_node` 的路由逻辑：

- `name` / `description` / `condition` / `exec_sql` → 走 `modify_outline` 管线（保留 L5 改名自动从 KB 同步等逻辑）
- 其余字段（`renderType` / `colX` / `colY` / `summarySuggestion` / `condition_queries`）→ patcher 的 `set_node_field` op 直接赋值

`set_outline` 收到 JSON 节点数组后经 `from_data()` 建树（与 LLM 上下文用的 `from_yaml` 共用建树逻辑），整棵替换当前大纲。

两者修改成功后都直接更新内存并推送 `outline` 事件（in-process，无需 session 文件中转）。

---

## Session 文件桥：内存 ↔ 脚本

bash 脚本运行在独立子进程，与 agent 内存不共享。两者通过会话文件
`/tmp/report_sessions/{session_id}.json`（可由 `REPORT_SESSION_DIR` 覆盖）桥接：

```
bash 调用前：把内存状态（outline_tree / outline_yaml / markdown / extraction）写入 session 文件
   ↓
脚本运行：读 session 文件 → 干活 → 把新状态写回 session 文件
   ↓
bash 调用后：读回 session 文件，_detect_events() 对比前后差异 →
             生成 outline / confirm / extraction / start_report 事件推给客户端
```

`_detect_events` 只在状态**真的变了**时才推事件（如 `outline_tree` 前后不等才推 `outline`）。这套机制让脚本无需感知 SSE / 调用方，只管读写 session 文件即可。

---

## Skill 系统：渐进式加载

| 级别 | 内容 | Token 代价 |
|------|------|-----------|
| Level 0 | skill 名称 + 一句话描述（启动时注入 system prompt） | 极少 |
| Level 1 | 完整 SOP（`read_skill(name)` 返回 SKILL.md 全文） | 中等，按需 |
| Level 2 | SOP 内引用的支持文件（`read_skill(name, path)`） | 按需 |

```
skills/
├── _lib/                # 脚本共享库（loader/retriever/subtree/patcher/outline_utils…）
├── analyze-network/     # 看网分析：检索→大纲→报告
│   ├── SKILL.md
│   └── scripts/         # search_graph_tree / build_outline_from_anchor / modify_outline / trigger_report …
├── consolidate-expert/  # 专家知识沉淀
└── publish-knowledge/   # 知识发布
```

`skill_registry.py` 启动时扫描 `skills/`，读取各 SKILL.md 的 YAML front matter 构成 Level 0 列表；`read_skill` 按需读取正文（Level 1）。System prompt（`system_prompt.txt`）要求 LLM 第一个动作必须是 `read_skill`，再严格按 SOP 执行。

---

## Memory：状态怎么管理

### 基类 AgentMemory（`memory/store.py`）

```python
class AgentMemory:
    outline_tree: dict    # 大纲 JSON（代码逻辑用）
    markdown: str         # 大纲 Markdown（用户视图，给人读）
    outline_yaml: str     # 大纲 YAML 精简视图（注入 LLM 上下文）
    kb_tree_text: str     # search_graph_tree 返回的树文本（暂存）
    _history: list[dict]  # 对话历史（不含大纲）
```

`AgentWithSkillsMemory` 在此基础上增加 `extraction`（场景元数据：scene_name / keywords / summary / usage_conditions）。

### 大纲为什么不存进对话历史？

大纲会随修改不断变化，存进历史会让旧版本一直残留、干扰 LLM，且每轮重复携带浪费 token。改为每次调 LLM 前由 `build_messages()` 把**最新** `outline_yaml` 动态拼入 system prompt：

```python
def build_messages(self, system_prompt):
    content = system_prompt
    if self.has_outline:
        content += f"\n\n## 当前大纲（可通过节点ID引用）\n\n{self.outline_yaml}"
    return [{"role": "system", "content": content}, *self._history]
```

> 拼进第一条 system 消息，而非追加新 system 消息——Qwen 等模型要求 system 只能在对话最开头。

### 大纲的三种格式

由 `skills/_lib/outline_utils.py` 从同一棵 `outline_tree` 派生：

| 格式 | 谁用 | 含 SQL 等字段 |
|------|------|--------------|
| `outline_tree`（JSON dict） | 代码逻辑（patcher 输入输出、报告执行器取数） | ✅ 完整 |
| `markdown` | 渲染给用户阅读 | ❌ |
| `outline_yaml` | 注入 LLM 上下文 | ❌ 省略 level/SQL 等，LLM 需要时用 `get_node_detail.py` 按需拉取 |

更详细的"为什么三份都不能省"见 `../docs/state-design.md`。

---

## 大纲修改：patcher.py

`modify_outline`（脚本）与 `edit_node`（工具）最终都调 `skills/_lib/patcher.apply_patch()` 执行一个 ops 列表：

| op | 参数 | 含义 |
|----|------|------|
| `add_node` | `node_id, parent_id[, after_id]` | 从知识图谱取节点（递归展开子树）插入指定父节点下 |
| `delete_node` | `node_id` | 删除节点及其全部子树 |
| `modify_node_name` | `node_id, value` | 改名；L5 节点会自动从 KB 同步 exec_sql/renderType 等关联字段 |
| `modify_node_description` | `node_id, value` | 改描述（仅 L1–L4） |
| `modify_node_condition` | `node_id, value` | 设展示条件（空串删除） |
| `modify_node_exec_sql` | `node_id, value` | 改 L5 的 exec_sql |
| `set_node_field` | `node_id, field, value` | 通用字段直接赋值（renderType/colX/colY 等），供 `edit_node` 使用 |
| `keep_only_node` | `node_id` | 保留该节点，同级其他全部删除 |

> 结构调整（增删/保留）走 `modify_outline.py`；改节点属性值优先走 `edit_node` 工具，详见 `skills/analyze-network/SKILL.md`。

---

## 报告生成与 mock 取数

报告由 `services/report_executor.py` 遍历大纲树、并行执行各 L5 指标的 SQL，通过 SSE 逐条推送（详见 `../docs/report-generation.md`）。取数有两个来源，由 `FORCE_MOCK` 决定优先级，且互相兜底：

```
FORCE_MOCK=false（默认，在线优先）：实时 SQL → 查空回落 mock
FORCE_MOCK=true （离线优先，无 DB 演示）：mock → 没有/失效回落实时 SQL
```

**mock 必须与当前 SQL 一致才复用**：`sql_executor.get_mock(node_id, current_sql)` 会比对「生成 mock 时所用的 SQL」（存于 `评估指标_mock.json` 每条记录的 `answer` 字段）与当前节点 exec_sql，归一化后不同则视为失效。这样改了 SQL 不会再套用旧指标的 mock，保证改 SQL 后重新取数。

离线 mock 由 `scripts/prefetch_mock_data.py` 预取生成。

---

## SSE 事件完整列表

**聊天回合（`/api/chat`，来自 `agent.chat_stream`）：**

```python
{"type": "step", "name": str, "status": "running", "call_id": str, "args": dict}   # 工具开始
{"type": "step", "name": str, "status": "done", "call_id": str,
 "result": str, "detail": str}                                                     # 工具结束（摘要+完整）
{"type": "outline", "markdown": str, "outline_yaml": str, "outline_tree": dict}    # 大纲更新（立刻推）
{"type": "confirm", "options": ["生成报告"]}                                        # 大纲变更后的确认选项
{"type": "extraction", "scene_name": str, "keywords": list, "summary": str}        # 场景元数据
{"type": "start_report"}                                                           # 触发报告渲染
{"type": "text", "chunk": str}                                                     # LLM 文字回复
{"type": "done", "seconds": float}                                                 # 本轮结束
{"type": "error", "message": str}                                                  # 出错
```

**报告渲染（`/api/report`，来自 `report_executor`）：**

```python
{"type": "report_metric",  "name": str, "chunk": str, "render_type"?, "rows"?, ...}  # 单条指标数据
{"type": "report_summary", "node_id": str, "chunk": str}                             # 节点总结（LLM 生成）
{"type": "report_skip",    "node_name": str}                                         # 条件不满足，跳过该节
{"type": "outline", ...}                                                             # 条件跳过后同步更新大纲
{"type": "report_done"}                                                              # 报告完成
```

---

## 目录结构

```
backend/
├── api_server.py              # FastAPI 入口（/api/chat、/api/report、/api/session…）
├── agent_with_skills/         # 唯一 Agent
│   ├── agent.py               # 主循环、工具分发、session 同步、_detect_events
│   ├── memory.py              # AgentWithSkillsMemory（+extraction）
│   ├── skill_registry.py      # 启动扫描 skills/，构建 Level 0 列表
│   ├── skill_loader.py        # 按需读取 SKILL.md 正文
│   ├── system_prompt.txt      # 系统提示词
│   └── agent_test.py          # 命令行交互测试
├── tools/shared_tools.py      # read_skill / bash / edit_node / set_outline 的 schema
├── skills/                    # 业务能力（SOP + 脚本 + _lib 共享库）
├── services/                  # llm_service / report_executor / sql_executor / de_sql_execution_client
├── memory/store.py            # AgentMemory 基类
├── llm/                       # LLM 配置
├── expert_knowledge/          # 知识库数据（node.json、评估指标_mock.json 等）
├── scripts/                   # 离线构建脚本（build_index、prefetch_mock_data、merge_… ）
└── tests/                     # test_sql_query / test_outline_utils / tool_test
```

---

## 环境变量

`.env` 文件（放在 `backend/` 目录下）：

| 变量 | 说明 | 示例 |
|------|------|------|
| `LLM_BASE_URL` | LLM API 地址（OpenAI 兼容） | `http://localhost:8000/v1` |
| `LLM_MODEL_NAME` | 模型名称 | `Qwen3-32B` |
| `LLM_API_KEY` | LLM API Key | `sk-xxx` |
| `LLM_TEMPERATURE` | 采样温度 | `0.1` |
| `LLM_TOP_P` / `LLM_TIMEOUT` / `LLM_ENABLE_THINKING` | 采样/超时/思考开关 | `1.0` / `120` / `false` |
| `EMBEDDING_BASE_URL` | Embedding 服务地址 | `http://localhost:8001/v1` |
| `EMBEDDING_DIM` | 向量维度 | `1024` |
| `FORCE_MOCK` | 离线优先（`true` 时优先用 mock 数据） | `false` |
| `REPORT_SESSION_DIR` | 会话文件目录 | `/tmp/report_sessions` |

> SQL 查询 API 的连接信息在 `config.yaml`（见 `config.example.yaml`）。

---

## 本地调试

### 启动服务

```bash
cd backend
uvicorn api_server:app --reload --port 8888
```

### 命令行交互测试（纯后端，无需 UI）

```bash
cd backend
python agent_with_skills/agent_test.py
```

内置命令：`/help`、`/skills`（列出可用 skill）、`/state`（大纲 JSON）、`/md`（Markdown 大纲）、`/reset`（清空历史与 skill 加载状态）。

### 离线构建向量索引

```bash
cd backend
python scripts/build_index.py
```

`expert_knowledge/node.json` 变更后需重建（首次启动若索引不存在也会自动构建）。

### 预取离线 mock 数据

```bash
cd backend
python scripts/prefetch_mock_data.py    # 批量执行评估指标 SQL，结果写入 评估指标_mock.json
```

### 单轮工具调用调试

```bash
cd backend
python tests/tool_test.py "帮我分析一下fgOTN的部署情况"
```

直接打印大模型原始 function calling 输出，不走完整 Agent 循环，用于排查工具定义或 prompt 问题。
