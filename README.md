# report_demo

AI 辅助的**报告大纲生成系统**。用户用自然语言描述分析需求，系统从知识库检索相关结构，生成可交互修改的报告大纲。专家也可以将自己的业务经验沉淀为可复用的大纲模板。

---

## 目录

- [项目功能](#项目功能)
- [技术栈](#技术栈)
- [项目结构](#项目结构)
- [核心概念](#核心概念)
- [三个 Agent](#三个-agent)
- [数据流：一次对话的完整链路](#数据流一次对话的完整链路)
- [启动方式](#启动方式)

---

## 项目功能

系统有两个主要使用场景：

| 场景 | 使用者 | 做什么 |
|------|--------|--------|
| **生成报告** | 普通用户 | 描述分析需求 → AI 检索知识库、生成大纲 → 对话式修改 |
| **沉淀专家知识** | 业务专家 | 输入业务经验 → AI 提炼结构 → 保存为可复用模板 |

---

## 技术栈

| 层 | 技术 |
|----|------|
| 前端 | React 18 + Vite，SSE 接收流式事件 |
| 后端 | Python + FastAPI，async 全程异步 |
| 大模型 | OpenAI 兼容接口（当前接入 Qwen），Function Calling |
| 向量检索 | FAISS，余弦相似度 |
| 向量化 | 自定义 embedding 服务 |

---

## 项目结构

```
report_demo/
├── frontend/                   # React 前端
│   └── src/
│       ├── pages/
│       │   ├── ChatView.jsx    # 主对话页面，处理所有 SSE 事件
│       │   ├── KBPage.jsx      # 知识库浏览页
│       │   └── TemplatePage.jsx# 模板浏览页
│       └── components/
│           ├── WorkflowSteps.jsx # 工具调用步骤展示
│           └── MarkdownOutline.jsx# 大纲渲染
│
└── backend/
    ├── api_server.py           # FastAPI 入口，session 管理，SSE 推流
    │
    ├── agent1/                 # 专家知识沉淀 Agent
    │   ├── agent.py            # 主循环（ReAct）
    │   ├── memory.py           # 状态管理（含模板元数据字段）
    │   ├── system_prompt.txt   # 系统提示词
    │   └── tools/              # 工具定义 + 调度（agent1 专用）
    │
    ├── agent2/                 # 报告大纲生成 Agent
    │   ├── agent.py            # 主循环（ReAct）
    │   ├── system_prompt.txt   # 系统提示词
    │   └── tools/              # 工具定义 + 调度（agent2 专用）
    │
    ├── agent_with_skills/      # 融合版 Agent（agent1 + agent2 工具合集）
    │   ├── agent.py            # 主循环，渐进式 skill 加载
    │   ├── prompt.txt          # 系统提示词
    │   ├── skill_registry.py   # 读取 skills/ 目录下的 SKILL.md 元数据
    │   └── skill_loader.py     # 加载 skill SOP 正文
    │
    ├── skills/                 # Skill SOP 定义（自然语言工作流）
    │   ├── generate-report/SKILL.md     # 报告生成流程
    │   └── consolidate-expert/SKILL.md  # 专家知识沉淀流程
    │
    ├── tools/                  # LLM 可调用工具（含 JSON Schema 定义）
    │   ├── shared_tools.py     # 所有 10 个工具的 Schema 定义（唯一来源）
    │   ├── search_graph_tree.py        # FAISS 检索知识库，返回带祖先路径的树
    │   ├── build_outline_from_anchor.py# 以锚节点为根展开子树生成大纲
    │   ├── search_template.py          # 向量检索模板候选
    │   ├── modify_outline.py           # 按 ops 列表修改大纲（纯 Python）
    │   ├── set_outline_from_markdown.py# 从 Markdown 渲染大纲（agent1 用）
    │   ├── set_scene_metadata.py       # 填写模板元数据
    │   └── save_template.py            # 持久化保存模板 JSON
    │
    ├── utils/                  # 内部基础设施（不暴露给 LLM）
    │   ├── loader.py           # 加载 FAISS 索引和知识图谱数据
    │   ├── retriever.py        # embedding + FAISS 检索 + 构建祖先路径
    │   ├── subtree.py          # 从知识图谱递归展开子树
    │   ├── patcher.py          # 执行大纲 patch 操作（add/delete/rename…）
    │   ├── template_selector.py# 模板向量检索
    │   └── outline_utils.py    # 大纲三种格式互转（tree / markdown / md_with_ids）
    │
    ├── services/
    │   ├── llm_service.py      # OpenAI 兼容客户端封装
    │   ├── embedding_service.py# 文本向量化
    │   └── faiss_service.py    # FAISS 索引读写
    │
    ├── expert_knowledge/
    │   ├── node.json           # 知识图谱节点（L1–L5 层级结构）
    │   └── relation.json       # 节点父子关系
    │
    ├── templates/              # 已保存的大纲模板（JSON 文件）
    └── scripts/
        └── build_index.py      # 离线构建 FAISS 向量索引
```

**`tools/` 和 `utils/` 的区别**

- `tools/`：每个文件对应一个 LLM 可调用的工具，`shared_tools.py` 里有完整的 JSON Schema 定义，是 agent 传给大模型的"工具说明书"
- `utils/`：纯内部基础设施，负责数据读取、向量运算、树操作，不直接暴露给 LLM

---

## 核心概念

### 1. Function Calling（工具调用）

大模型原生只能输出文字。Function Calling 是 OpenAI 协议扩展，允许大模型表达"我想调用某个函数"。

调用 API 时传入 `tools` 参数（JSON Schema 格式的工具说明），大模型在合适时机不输出文字，而是输出：

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

代码解析这个结构，执行对应工具，把结果放回对话历史，再调一次 LLM，如此循环直到大模型输出普通文字为止。这个模式叫 **ReAct**（Reason + Act）。

### 2. ReAct 主循环

每个 agent 的 `chat_stream()` 都是同一个模式：

```
调 LLM → 工具调用? → 执行工具 → 结果放回历史 → 调 LLM → ... → 普通回复 → 结束
```

见 `backend/agent2/agent.py` 的 `chat_stream()` 方法，逻辑清晰，约 50 行。

### 3. SSE 事件协议

`chat_stream()` 是 async generator，每发生一件事就 `yield` 一个字典。`api_server.py` 把它们序列化成 Server-Sent Events 推给前端：

| 事件 | 含义 | 关键字段 |
|------|------|---------|
| `step` running | 工具开始执行 | `name`, `call_id`, `args` |
| `step` done | 工具执行完毕 | `name`, `call_id`, `result`, `detail` |
| `outline` | 大纲更新，前端立刻渲染 | `markdown`, `outline_tree` |
| `extraction` | 元数据提取完成（agent_with_skills 专用） | `scene_name`, `keywords`, `summary` |
| `saved` | 模板保存成功 | `scene_name`, `path` |
| `text` | LLM 文字回复 | `chunk` |
| `done` | 本轮结束 | `seconds` |
| `error` | 出错 | `message` |

大纲走独立的 `outline` 事件而不是让 LLM 逐字输出，因为大纲是工具**计算出来的**，无需 LLM 重新生成，前端可以瞬间渲染。

### 4. 大纲的三种格式

大纲在系统中以三种形态存在，服务不同目的：

| 格式 | 示例 | 用途 |
|------|------|------|
| `outline_tree` | JSON dict，含 id/name/level/children | 代码逻辑（`modify_outline` 的输入输出） |
| `markdown` | `# fgOTN部署\n## 传送网分析` | 前端渲染，给用户看 |
| `md_with_ids` | `[L2 L2_001] fgOTN部署\n  [L3 L3_001] …` | 注入 LLM 上下文，让大模型能精确引用节点 ID |

三者由 `utils/outline_utils.py` 从同一棵 `outline_tree` 派生。

### 5. Memory（会话状态）

大纲不存入对话历史（避免旧版本误导 LLM、浪费 token），而是存在 `memory` 对象里。每次调 LLM 前，`build_messages()` 把当前最新的 `md_with_ids` 动态拼入 system prompt 末尾。

---

## 三个 Agent

系统有三个 agent，通过 `POST /api/session { "agent_id": 1|2|3 }` 选择：

### Agent1（专家知识沉淀）

工具链：`search_graph_tree → set_outline_from_markdown → set_scene_metadata → modify_outline → save_outline_template`

专家输入业务经验描述，Agent1 检索知识图谱、构建大纲、填写元数据，最终保存为 JSON 模板供后续复用。

### Agent2（报告大纲生成）

工具链（有模板）：`search_outline_templates → load_template_outline`  
工具链（无模板）：`search_graph_tree → build_outline_from_anchor → modify_outline`

普通用户提需求，Agent2 优先匹配已有模板，否则从知识库实时构建大纲，支持对话式修改。

### AgentWithSkills（融合版，推荐使用）

合并了 Agent1 + Agent2 的全部工具，通过 **渐进式 Skill 加载** 动态选择工作流：

- 启动时只向 LLM 注入 skill 名称和简短描述（Level 0，极少 token）
- LLM 判断需要时调用 `read_skill(name)` 获取完整 SOP（Level 1）
- SOP 是自然语言写的工作流指导，告诉 LLM 该调哪些工具、按什么顺序

这种设计让单个 agent 同时支持两种场景，且 token 消耗与 Agent1/Agent2 相当。

---

## 数据流：一次对话的完整链路

```
用户输入
  │
  ▼
前端 POST /api/chat
  │
  ▼
api_server.py → 从 _sessions 取出对应 agent
  │
  ▼
agent.chat_stream(message)          ← async generator
  │
  ├─ 调 LLM（携带 TOOLS + 对话历史）
  │
  ├─ LLM 返回 tool_calls
  │   ├─ yield {"type":"step","status":"running",...}  ── SSE ──▶ 前端显示"执行中"
  │   ├─ 执行工具（tools/ + utils/）
  │   ├─ yield {"type":"outline",...}                 ── SSE ──▶ 前端立刻渲染大纲
  │   └─ yield {"type":"step","status":"done",...}    ── SSE ──▶ 前端标记"完成"
  │
  ├─ 把工具结果放回 messages，再次调 LLM
  │
  └─ LLM 返回普通文字
      ├─ yield {"type":"text","chunk":...}            ── SSE ──▶ 前端显示回复
      └─ yield {"type":"done","seconds":...}          ── SSE ──▶ 本轮结束
```

---

## 启动方式

### 前端

```bash
cd frontend
npm install
npm run dev          # 默认 http://localhost:5173
```

### 后端

```bash
cd backend
cp .env.example .env  # 填写 LLM_BASE_URL / LLM_API_KEY / EMBED_BASE_URL 等
uvicorn api_server:app --reload --port 8000
```

### 离线构建向量索引（首次或知识库变更后）

```bash
cd backend
python scripts/build_index.py
```

### 命令行交互测试（无需启动前端）

```bash
cd backend
python agent2/agent_test.py           # 测试 Agent2
python agent_with_skills/agent_test.py # 测试 AgentWithSkills
```
