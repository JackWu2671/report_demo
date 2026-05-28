# report_demo

AI 驱动的网络分析报告生成系统。用户用自然语言描述分析需求，Agent 从知识库检索相关节点、生成可对话修改的报告大纲，并执行 SQL 查询填充数据，最终生成含图表的 Markdown 报告。

---

## 快速启动

### 前置条件

- Python 3.10+
- Node.js 18+
- 可访问的 LLM 服务和 Embedding 服务

### 1. 配置

```bash
# 后端环境变量
cp backend/.env.example backend/.env
# 编辑 .env，填写 LLM_BASE_URL、EMBEDDING_BASE_URL 等

# SQL 查询 API 配置（可选，不填则使用 mock 数据）
cp backend/config.example.yaml backend/config.yaml
# 编辑 config.yaml，填写 base_url、operator、user_id 等
```

### 2. 启动后端

```bash
cd backend
pip install -r requirements.txt
uvicorn api_server:app --host 0.0.0.0 --port 8888 --reload
```

首次启动会自动构建 FAISS 索引（需要 Embedding 服务可用），之后直接加载。

### 3. 启动前端

```bash
cd frontend
npm install
npm run dev
```

默认在 `http://localhost:5173` 启动，API 请求代理到 `http://localhost:8888`。

---

## 目录结构

```
report_demo/
├── frontend/                  # React 前端
├── backend/                   # Python 后端
│   ├── api_server.py          # FastAPI 入口
│   ├── agent_with_skills/     # Agent 核心
│   ├── skills/                # 技能包（SOP + 脚本）
│   ├── services/              # 基础服务
│   ├── expert_knowledge/      # 知识库数据
│   ├── scripts/               # 离线构建脚本
│   ├── tests/                 # 测试
│   ├── .env.example           # 环境变量模板
│   └── config.example.yaml    # SQL 查询 API 配置模板
└── docs/                      # 技术文档
```

---

## 前端（`frontend/`）

```
frontend/src/
├── App.jsx                    # 路由与布局
├── pages/
│   ├── ChatView.jsx           # 主对话页：骨架构建、SSE 事件处理、报告生成
│   ├── ChatPage.jsx           # 对话页容器
│   ├── KBPage.jsx             # 知识库浏览页
│   └── TemplatePage.jsx       # 模板库浏览页
└── components/
    ├── ReportView.jsx         # 报告渲染：Markdown + ECharts 图表 + TOC 导航
    ├── ChatMessage.jsx        # 对话消息气泡
    ├── MarkdownOutline.jsx    # 大纲展示组件
    ├── QueryInput.jsx         # 输入框
    ├── TreeNode.jsx           # 知识树节点
    └── WorkflowSteps.jsx      # 工作流步骤指示
```

**主要依赖：**

| 包 | 用途 |
|----|------|
| `react-markdown` | 把 Markdown 字符串渲染成 React 组件 |
| `remark-gfm` | 支持 GFM 语法（表格等） |
| `rehype-raw` | 允许 Markdown 中的 HTML 标签真正渲染 |
| `echarts-for-react` | ECharts 图表组件（BAR / LINE / PIE） |

---

## 后端（`backend/`）

### `api_server.py`

FastAPI 服务入口，提供以下接口：

| 接口 | 说明 |
|------|------|
| `POST /api/session` | 创建对话会话 |
| `POST /api/chat` | 对话（SSE 流式响应） |
| `POST /api/report` | 生成报告（SSE 流式响应） |
| `GET  /api/kb` | 知识库节点与关系 |
| `GET  /api/templates` | 模板列表 |
| `GET  /api/session/{id}/messages` | 会话历史 |

服务器启动时通过 `lifespan` 钩子检查 FAISS 索引，缺失时自动构建。

---

### `agent_with_skills/`

Agent 核心，负责驱动 LLM 与技能包交互。

```
agent_with_skills/
├── agent.py          # Agent 主循环（chat_stream、工具分发、session 同步）
├── memory.py         # 会话状态管理（对话历史、大纲、场景元数据）
├── skill_loader.py   # 启动时加载所有 skills 的名称与描述
├── skill_registry.py # Skill 注册表
└── system_prompt.txt # 系统提示词基础模板
```

LLM 只感知两个工具：`read_skill`（读取 SOP）和 `bash`（执行脚本）。

---

### `skills/`

技能包：每个子目录是一个独立技能，包含 `SKILL.md`（工作流 SOP）和 `scripts/`（业务脚本）。

```
skills/
├── _lib/                          # 脚本共享库
│   ├── loader.py                  # 加载 FAISS 索引和知识图谱
│   ├── retriever.py               # 向量检索：embed → search → 路径补全 → 树
│   ├── patcher.py                 # 大纲 patch 操作执行器
│   ├── outline_utils.py           # 大纲三种格式互转（JSON / Markdown / md_with_ids）
│   ├── subtree.py                 # 从知识图谱展开子树
│   ├── template_selector.py       # 模板实时向量检索
│   ├── session.py                 # 读写 session 文件（agent ↔ 脚本的数据桥梁）
│   ├── build_outline_from_anchor.py
│   ├── modify_outline.py
│   ├── set_outline_from_markdown.py
│   ├── search_graph_tree.py
│   ├── search_template.py
│   ├── save_template.py
│   └── set_scene_metadata.py
│
├── analyze-network/               # 看网分析技能
│   ├── SKILL.md                   # SOP：检索 → 构建大纲 → 对话修改 → 生成报告
│   └── scripts/
│       ├── search_graph_tree.py
│       ├── build_outline.py
│       ├── modify_outline.py
│       ├── load_template.py
│       ├── search_templates.py
│       └── trigger_report.py      # 对话触发报告生成的信号脚本
│
├── consolidate-expert/            # 专家知识沉淀技能
│   ├── SKILL.md                   # SOP：提取大纲 → 填写元数据 → 保存模板 → 融合知识图谱
│   └── scripts/
│       ├── set_outline.py
│       ├── set_metadata.py
│       ├── save_template.py
│       └── graph_manage.py
│
└── publish-knowledge/             # 知识发布技能
    ├── SKILL.md
    └── scripts/
        ├── graph_manage.py
        ├── list_templates.py
        └── show_graph.py
```

---

### `services/`

底层服务，不含业务逻辑。

```
services/
├── embedding_service.py       # Embedding HTTP 客户端（单条 / 批量向量化）
├── faiss_service.py           # FAISS 向量索引（build / save / load / search）
├── llm_service.py             # LLM HTTP 客户端
├── report_executor.py         # 报告生成调度：遍历大纲树，并行执行 SQL，SSE 推送
├── sql_executor.py            # 指标名 → SQL 执行，mock_data 离线降级
└── de_sql_execution_client.py # SQL 查询 API 客户端（POST 提交任务 → GET 轮询结果）
```

---

### `expert_knowledge/`

知识库静态数据，由 `scripts/` 下的构建脚本生成，提交到 git。

```
expert_knowledge/
├── node.json          # 知识节点列表（id、name、level、keywords 等）
├── relation.json      # 父子关系列表（parent、child、order）
└── 评估指标_mock.json  # 指标 SQL 定义及离线 mock 数据
```

FAISS 索引文件（`data/faiss.index`、`data/faiss_id_map.json`）不提交 git，运行时自动生成。

---

### `scripts/`

离线一次性脚本，不参与运行时。

```
scripts/
├── build_index.py              # 手动构建 FAISS 向量索引
├── build_knowledge_nodes.py    # 从 Excel 生成 node.json
├── build_knowledge_relations.py # 从 Excel 生成 relation.json
├── parse_scene_xlsx.py         # 解析场景 Excel
├── parse_evaluation_item_xlsx.py # 解析评估项 Excel
├── merge_sample_questions.py   # 合并示例问题
└── prefetch_mock_data.py       # 离线预取 SQL 结果，写入 mock_data 字段
```

---

### `tests/`

```
tests/
├── test_sql_query.py    # 独立 SQL 查询测试（无项目依赖，填写 CONFIG 直接运行）
├── test_outline_utils.py
└── tool_test.py
```

---

## 配置

### 1. 环境变量

```bash
cp backend/.env.example backend/.env
# 填写 LLM_BASE_URL、EMBEDDING_BASE_URL 等
```

### 2. SQL 查询 API

```bash
cp backend/config.example.yaml backend/config.yaml
# 填写 base_url、operator、user_id 等
```

---

## 技术文档

详细流程说明见 `docs/`：

| 文档 | 内容 |
|------|------|
| `docs/report-generation.md` | 从大纲 JSON 到最终报告的完整流程 |
| `docs/index-and-retrieval.md` | FAISS 索引构建、自动检测与知识检索流程 |
