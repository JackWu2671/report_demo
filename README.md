# report_demo

AI 辅助的报告大纲生成系统。用户用自然语言描述分析需求，系统从知识库检索相关节点、生成可对话修改的报告大纲；专家也可以把自己的业务经验沉淀为可复用的大纲模板。

---

## 目录

1. [报告的三种表现形式](#1-报告的三种表现形式)
2. [知识库存储](#2-知识库存储)
3. [模板存储](#3-模板存储)
4. [知识库检索](#4-知识库检索)
5. [模板检索](#5-模板检索)
6. [Agent Loop 设计](#6-agent-loop-设计)
7. [Tools 设计](#7-tools-设计)
8. [Skills 设计](#8-skills-设计)
9. [Memory 设计](#9-memory-设计)

---

## 1. 报告的三种表现形式

同一份报告大纲在系统中以三种形态并存，服务不同的消费者。三者均由 `utils/outline_utils.py` 从同一个 `outline_tree` 派生，互不依赖。

### 1.1 outline_tree（结构化 JSON）

程序逻辑的唯一数据源。所有工具的输入输出都以此格式传递，存储在 `memory.outline_tree`。

```json
{
  "id": "__root__",
  "name": "",
  "level": 0,
  "description": "",
  "children": [
    {
      "id": "L1_001",
      "name": "政企OTN升级",
      "level": 1,
      "description": "引导fgOTN部署，推荐部署的站点",
      "condition": "",
      "children": [
        {
          "id": "L2_001",
          "name": "fgOTN部署",
          "level": 2,
          "description": "",
          "children": [
            {
              "id": "L5_001",
              "name": "企业行业分布",
              "level": 5,
              "description": "仅统计南宁市的企业行业分布",
              "children": []
            }
          ]
        }
      ]
    }
  ]
}
```

**虚拟根节点**：`id = "__root__"` 的顶层节点是系统内部节点，不对应实际内容，用于让 `add_node` 操作在顶层章节平行插入时有统一的 `parent_id`（传空字符串）。

**节点字段说明：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 知识库原始 ID（如 `L4_002`）或新建节点 ID（如 `new_001`） |
| `level` | int | 层级深度，1～4 为结构节点，5 为 query 节点 |
| `description` | string | L1～L4 为说明性文字；**query 节点的 description 直接作为查询参数执行数据过滤** |
| `condition` | string | 展示条件，如"当用户选择了城市时，本节才展示" |
| `children` | array | 子节点列表；query 节点（level=5）必须为空 |

### 1.2 markdown（用户视图）

纯 Markdown，去掉所有 ID，供前端渲染给用户查看。由 `to_markdown(tree)` 生成。

```markdown
# 政企OTN升级

引导fgOTN部署，推荐部署的站点

## fgOTN部署

### 传送网络覆盖分析

#### 企业分布分析

从行业、行政区等维度统计企业分布

##### 企业行业分布

仅统计南宁市的企业行业分布
```

渲染规则：树的第一层对应 `#`，每深一层加一级标题（最深 `######`）；`description` 渲染为标题下方段落；`condition` 渲染为 `@if ...` 标记。

### 1.3 md_with_ids（LLM 上下文视图）

带节点 ID 的缩进树格式，每轮 LLM 调用前注入 system prompt，让大模型能精确引用节点 ID 来构造修改操作。由 `to_markdown_with_ids(tree)` 生成。

```
[L1 L1_001] 政企OTN升级：引导fgOTN部署，推荐部署的站点
  [L2 L2_001] fgOTN部署
    [L3 L3_001] 传送网络覆盖分析
      [L4 L4_001] 企业分布分析：从行业、行政区等维度统计企业分布
        [Q L5_001] 企业行业分布：仅统计南宁市的企业行业分布
        [Q L5_002] 企业行政区分布
```

**格式规则：**

- `[L{n} {id}]` 表示结构节点（L1～L4）；`[Q {id}]` 表示 query 节点（level=5）
- 名称后紧跟全角冒号 `：` 和 description（有则显示，无则省略）
- 条件节点额外追加 `｜条件：当……时，本节才展示`
- 新建节点使用 `new_xxx` 作为 ID，知识库原有节点保留原始 ID

**为什么 query 节点用 `[Q ...]` 而不是 `[L5 ...]`：** query 节点是叶子节点，背后对应一条 SQL 或 API 查询，语义上与 L1～L4 的结构节点不同。`Q` 前缀让 LLM 一眼识别哪些节点是数据查询入口，避免在其下挂子节点或误操作。

---

## 2. 知识库存储

知识库以两个 JSON 文件存储在 `backend/expert_knowledge/`，启动时由 `utils/loader.py` 一次性加载到内存。

### 2.1 node.json

节点列表，每个节点一个对象：

```json
[
  {
    "id": "L1_001",
    "level": 1,
    "name": "政企OTN升级",
    "keywords": ["OTN", "政企", "升级", "传送网", "fgOTN", "量子加密"],
    "description": "引导fgOTN部署，推荐部署的站点，针对这部分站点可以优先引导部署细颗粒功能板"
  },
  {
    "id": "L5_001",
    "level": 5,
    "name": "企业行业分布",
    "keywords": ["企业", "行业", "分布"],
    "description": "按行业分类统计目标区域内的企业数量和占比"
  }
]
```

**当前知识库规模：** 24 个节点，层级分布：

| Level | 含义 | 数量 |
|-------|------|------|
| L1 | 顶层主题 | 1 |
| L2 | 业务方向 | 2 |
| L3 | 分析域 | 2 |
| L4 | 分析模块 | 5 |
| L5（Query） | 数据查询入口 | 14 |

**当前完整树结构：**

```
[L1 L1_001] 政企OTN升级
  [L2 L2_001] fgOTN部署
    [L3 L3_001] 传送网络覆盖分析
      [L4 L4_001] 企业分布分析
        [Q L5_001] 企业行业分布
        [Q L5_002] 企业行政区分布
        [Q L5_003] 企业城市分布
        [Q L5_004] 企业详情
      [L4 L4_002] 企业覆盖分析
        [Q L5_005] OTN站点企业覆盖率
        [Q L5_006] OTN站点覆盖企业数量
        [Q L5_007] OTN站点覆盖企业行政区域分布
        [Q L5_008] OTN站点未覆盖企业行政区域分布
    [L3 L3_002] 传送网络容量分析
      [L4 L4_003] 低阶交叉资源分析
        [Q L5_009] 站点低阶交叉容量利用率区间分布
        [Q L5_010] 站点设备低阶交叉容量使用详情
      [L4 L4_004] 设备槽位资源分析
        [Q L5_011] 子架槽位利用率区间分布
        [Q L5_012] 子架业务槽位使用详情
      [L4 L4_005] 站点fgOTN支持度分析
        [Q L5_013] 站点支持部署fgOTN单板的子架详情
        [Q L5_014] 站点支持部署fgOTN单板的状态分布
  [L2 L2_002] 量子加密板部署
    [L3 L3_001] 传送网络覆盖分析
      （与 fgOTN 部署共享相同的覆盖分析节点）
```

### 2.2 relation.json

父子关系列表，每条记录一个父子对：

```json
[
  { "parent": "L1_001", "child": "L2_001", "order": 1 },
  { "parent": "L1_001", "child": "L2_002", "order": 2 },
  { "parent": "L2_001", "child": "L3_001", "order": 1 }
]
```

`order` 字段决定同级子节点的显示顺序。`loader.py` 将其处理为 `children_map: {parent_id → [child_id, ...]}` 字典，供 `subtree.py` 和 `retriever.py` 使用。

### 2.3 FAISS 向量索引

节点向量化对象为 `name + keywords + description` 拼接文本，离线构建：

```bash
python scripts/build_index.py
```

索引文件存储在 `backend/` 目录，启动时由 `services/faiss_service.py` 载入内存。

---

## 3. 模板存储

每个保存的大纲模板是 `backend/templates/` 目录下的一个独立 JSON 文件，文件名为 UUID。

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "scene_name": "fgOTN覆盖评估",
  "summary": "评估OTN站点对目标企业的覆盖现状，识别覆盖缺口",
  "keywords": ["OTN", "覆盖率", "企业分布", "传送网", "fgOTN"],
  "usage_conditions": "适用于有OTN网络现状数据、需要评估企业覆盖缺口的场景",
  "created_at": "2024-01-15 14:30:00",
  "outline": {
    "id": "__root__",
    "name": "",
    "level": 0,
    "description": "",
    "children": [ ... ]
  }
}
```

**字段说明：**

| 字段 | 来源 | 用途 |
|------|------|------|
| `id` | 保存时自动生成 UUID | 唯一标识，供 `load_template_outline` 按 id 加载 |
| `scene_name` | 专家填写，≤10字 | 检索文本的一部分；向用户展示 |
| `summary` | 专家填写，≤50字 | 检索文本的一部分；帮助 LLM 判断相关性 |
| `keywords` | 专家填写，3～8个 | 不参与向量检索，供人工浏览 |
| `usage_conditions` | 专家填写，≤80字 | 检索文本的一部分；描述适用前提 |
| `created_at` | 保存时自动写入 | 记录创建时间 |
| `outline` | 当前 `outline_tree` | 完整大纲结构（与 node.json 格式一致） |

**检索文本拼接：** `scene_name + " " + summary + " " + usage_conditions`，用于模板向量检索。

---

## 4. 知识库检索

实现在 `utils/retriever.py`，由 `tools/search_graph_tree.py` 封装为 LLM 可调工具。

### 检索流程

```
用户问题（自然语言）
    │
    ▼ embed_query()
问题向量（shape: 1×dim，已归一化）
    │
    ▼ search_nodes() — FAISS 余弦相似度，top-10，过滤 score < 阈值
命中节点列表
    │
    ▼ build_candidate_paths()
带祖先路径的候选节点（path = "政企OTN升级 > fgOTN部署 > ..."）
    │
    ▼ 展开命中节点的所有后代（避免因阈值过滤漏掉 query 节点）
完整子树
    │
    ▼ _tree_to_text()
树状文本（md_with_ids 格式）供 LLM 选锚节点
```

### 关键参数

| 参数 | 默认值 | 配置方式 |
|------|--------|---------|
| top-k | 10 | 代码常量 |
| score 阈值 | 0.3 | 环境变量 `FAISS_SCORE_THRESHOLD` |
| embedding 维度 | 1024 | 环境变量 `EMBEDDING_DIM` |

### 返回给 LLM 的格式

```
[L2 L2_001] fgOTN部署：引导fgOTN部署，推荐部署的站点
  [L3 L3_001] 传送网络覆盖分析
    [L4 L4_001] 企业分布分析：从行业、行政区等维度统计企业分布
      [Q L5_001] 企业行业分布：按行业分类统计目标区域内的企业数量
      [Q L5_002] 企业行政区分布
```

LLM 从这棵树中选择锚节点 ID 传给 `build_outline_from_anchor`，或直接引用 query 节点 ID 构造大纲（`consolidate-expert` 场景）。

---

## 5. 模板检索

实现在 `utils/template_selector.py`，由 `tools/search_template.py` 封装。

### 检索流程

```
用户问题（自然语言）
    │
    ▼ embed_query()
问题向量（shape: 1×dim）
    │
    ▼ _load_templates() — 实时读取 templates/*.json（每次检索时读盘）
所有模板的检索文本 = scene_name + summary + usage_conditions
    │
    ▼ get_embeddings_batch() — 批量向量化（无预建索引，实时计算）
模板向量矩阵（shape: N×dim，已归一化）
    │
    ▼ template_vecs @ query_vec.T → 余弦相似度（N,）
    │
    ▼ argsort 降序，取 top-K
候选模板列表（含 id、scene_name、summary、score）
```

**与知识库检索的区别：** 知识库使用预建 FAISS 索引（O(log N) 查询）；模板检索在每次调用时实时计算向量（O(N)）。模板数量少（几十到几百），实时计算代价可接受，且不需要维护离线索引文件。

### 返回给 LLM 的格式

```
[search_outline_templates] 找到 3 个候选:
  1. id=550e8400...  scene_name=fgOTN覆盖评估  score=0.923
      summary: 评估OTN站点对目标企业的覆盖现状
      usage_conditions: 适用于有OTN网络现状数据的场景
  2. ...
```

LLM 根据 `scene_name`/`summary`/`usage_conditions` 自行判断相关性，有匹配则调 `load_template_outline`，否则转向 `search_graph_tree`。

---

## 6. Agent Loop 设计

核心实现在 `agent_with_skills/agent.py` 的 `chat_stream()` 方法。

### 主循环结构

```python
async def chat_stream(user_message):
    memory.add_message({"role": "user", "content": user_message})

    for _ in range(MAX_ROUNDS):           # 最多 8 轮工具调用
        response = await _call_llm()      # 携带完整对话历史 + 工具列表

        if finish_reason == "tool_calls":
            for tc in tool_calls:
                yield step_running_event       # 推给前端：工具开始执行
                result_dict, llm_str = await _execute_tool(tc)
                yield outline/extraction/saved_events  # 按需推送业务事件
                yield step_done_event          # 推给前端：工具执行完毕
                memory.add_message(tool_result)
            continue                           # 继续下一轮

        yield text_event                       # LLM 文字回复
        yield done_event
        return

    yield error_event("工具调用次数超限")
```

### 工具分发（dispatch map）

```python
async def _execute_tool(tool_call):
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)  # OpenAI 格式，参数是 JSON 字符串

    if name == "read_skill":
        return _handle_read_skill(args)   # skill 元工具，agent 本地处理

    handler = HANDLERS.get(name)          # 业务工具，查 dispatch map
    return await handler(args, self.memory)
```

### SSE 事件协议

| 事件类型 | 触发时机 | 关键字段 |
|----------|----------|---------|
| `step` running | 工具开始执行 | `name`, `call_id`, `args` |
| `step` done | 工具执行完毕 | `name`, `result`（单行摘要）, `detail`（完整 llm_str）|
| `outline` | 任何产生新大纲的工具返回后 | `markdown`, `md_with_ids`, `outline_tree` |
| `extraction` | `set_scene_metadata` 成功后 | `scene_name`, `keywords`, `summary` |
| `saved` | `save_outline_template` 成功后 | `scene_name`, `path` |
| `text` | LLM 文字回复 | `chunk` |
| `done` | 本轮结束 | `seconds` |
| `error` | 超限或异常 | `message` |

大纲走独立的 `outline` 事件而不是让 LLM 逐字输出，因为大纲是工具计算出来的结构化数据，无需 LLM 重新生成，前端可以瞬间渲染。

### 工具结果的双返回值

每个 handler 返回 `(result_dict, llm_str)` 两部分：

- **`result_dict`**：结构化 JSON，agent loop 检查它决定推哪种 SSE 事件（`outline_tree` 是否非空、`status` 是否 success）
- **`llm_str`**：紧凑文本，写入 LLM 对话历史，格式如 `[tool_name] status=success\n...`

两者分离是因为前端需要结构化 JSON，而 LLM 需要自然语言摘要，同一份执行结果服务两个消费者。

---

## 7. Tools 设计

所有工具的 JSON Schema 定义统一存放在 `tools/shared_tools.py`（OpenAI Function Calling 格式），handler 实现在对应的 `tools/*.py` 文件中。当前共 9 个工具，分三组。

### 7.1 Skill 元工具（1个）

#### `read_skill`

加载指定 skill 的完整 SOP 或其内部支持文件，由 `agent.py` 直接处理，不走 dispatch map。

```
参数：
  name  (string, 必填) — skill 名称，如 analyze-network
  path  (string, 选填) — skill 目录内的支持文件路径（Level 2 加载）

返回（写入 LLM 历史）：
  Level 1（path 为空）：[read_skill Level 1] {name}:\n\n{SKILL.md 正文}
  Level 2（path 非空）：[read_skill Level 2] {name}/{path}:\n\n{文件内容}
```

已加载的 SOP 通过 `_loaded` set 去重，同一会话内不重复注入（返回"已加载，请直接按流程操作"）。内置路径沙箱，`ref_path` 经 `.resolve().is_relative_to()` 验证，防止读取 skill 目录之外的文件。

### 7.2 知识库与大纲构建工具（3个）

#### `search_graph_tree`

从知识图谱检索与问题相关的节点，返回带祖先路径的树状结构。

```
参数：question (string) — 用户需求原文

返回：
  status="success"   : {tree_text（md_with_ids 格式文本）, graph_tree（JSON）}
  status="not_found" : {message}
```

LLM 从 `tree_text` 中选锚节点 ID 传给 `build_outline_from_anchor`，或引用 query 节点 ID 直接构造大纲。

#### `build_outline_from_anchor`

以指定节点为根，展开知识图谱子树，生成初始大纲。

```
参数：anchor_id (string) — 从 search_graph_tree 返回结果中选取的节点 ID

返回：
  status="success"   : {outline_tree, markdown, md_with_ids}
  status="not_found" : {message}
```

将选中节点下的所有后代节点完整展开，生成的大纲通常比实际需要的更宽，需随后调用 `modify_outline` 修剪。

#### `set_outline_from_markdown`

将 LLM 自行构造的 `md_with_ids` 格式文本解析为结构化大纲，用于 `consolidate-expert` 场景。

```
参数：md_with_ids (string) — 符合格式约束的大纲文本

返回：
  status="success" : {outline_tree, markdown, md_with_ids}
  status="error"   : {message}
```

格式约束（违反任意一条为无效输出）：
- L1 必须存在且唯一，作为报告总标题
- query 节点（`[Q ...]`）必须是叶子节点，禁止有子节点
- 禁止新建 query 节点，只能引用知识库已有节点 ID
- 每个新建 L4 下必须至少挂一个知识库已有的 query 节点
- 所有新建节点名称后必须紧跟全角冒号和 50～100 字描述

### 7.3 模板管理工具（2个）

#### `search_outline_templates`

向量检索模板库，返回 top-N 候选列表。

```
参数：question (string), top_k (int, 默认5)

返回：
  status="found"     : {candidates: [{id, scene_name, summary, usage_conditions, score}]}
  status="not_found" : {reason}
```

LLM 根据候选的 `scene_name`/`summary`/`usage_conditions` 自行判断相关性，有匹配则调 `load_template_outline`，无匹配则转向 `search_graph_tree`。

#### `load_template_outline`

按模板 ID 直接加载完整大纲（O(1) 文件读取）。

```
参数：template_id (string) — 来自 search_outline_templates 返回的 id 字段

返回：
  status="success"   : {outline_tree, markdown, md_with_ids, scene_name}
  status="not_found" : {reason}
```

内置路径沙箱验证（`template_id` 拼路径后经 `.resolve().is_relative_to(TEMPLATE_DIR)` 验证），防止路径遍历攻击。

### 7.4 大纲修改与保存工具（3个）

#### `modify_outline`

对当前大纲执行结构化修改，支持多操作批量执行，按列表顺序依次应用。

```
参数：ops (array) — 操作列表

支持的操作：
  add_node               {op, node_id, parent_id}  从知识库添加已有节点；parent_id="" 表示顶层
  delete_node            {op, node_id}             删除节点及其整个子树
  modify_node_name       {op, node_id, value}      修改节点名称
  modify_node_description{op, node_id, value}      修改节点描述（query 节点直接影响数据查询范围）
  modify_node_condition  {op, node_id, value}      设置/删除展示条件；value="" 表示删除
  keep_only_node         {op, node_id}             保留该节点，删除同级所有其他节点

返回：{status, outline_tree, markdown, md_with_ids, ops（已执行）, skipped（跳过的操作及原因）}
```

`skipped` 字段非空时 LLM 必须继续调工具补救，不得直接告知用户已完成。

#### `set_scene_metadata`

填写场景元数据，存入 `memory.extraction`，供后续保存模板使用。

```
参数：scene_name (≤10字), summary (≤50字), keywords (3～8个), usage_conditions (≤80字)
返回：{status, scene_name, summary, keywords, usage_conditions}
```

#### `save_outline_template`

将 `memory.extraction` + `memory.outline_tree` 持久化为 JSON 模板文件。

```
参数：（无，直接从 memory 读取）
返回：{status, path, scene_name, template_id}
```

只在用户明确确认时调用（说"保存"、"好的就这样"等），不得主动触发。

---

## 8. Skills 设计

Skills 是用自然语言写的工作流 SOP，告诉 LLM 面对特定场景时应该按什么顺序调用哪些工具。存放在 `backend/skills/` 目录，每个 skill 是一个子目录，包含 `SKILL.md`（主 SOP）和可选的支持文件。

### 三级渐进式加载

```
Level 0（系统启动时缓存）
  skill 名称 + 描述注入 system prompt（约 100 token/skill）
  LLM 从此知道有哪些能力可用

          ↓ LLM 判断需要某个 skill，调用 read_skill(name)

Level 1（按需加载，每 skill 约 2000 token）
  完整 SKILL.md SOP 正文通过 tool_result 注入对话历史
  LLM 严格按 SOP 执行后续工具调用

          ↓ SOP 中引用支持文档，调用 read_skill(name, path)

Level 2（按需加载）
  skill 目录内的指定支持文件（如 node-text-format.md）
```

**去重机制：** 已加载的 SOP 记录在 `agent._loaded` set 中，同一会话内不重复注入，避免浪费 context。

**Level 0 注入位置（在 system prompt 内）：**
```
<skill_system>
调用工具时，遇到复杂任务先用 read_skill(<skill_name>) 阅读工作流指导。
<available_skills>
- [report] analyze-network: 看网分析工具包。用于一切需要分析传送网络现状的场景...
- [report] consolidate-expert: 专家知识沉淀工具包。用户发来一段较长的业务描述...
</available_skills>
</skill_system>
```

### 8.1 analyze-network（看网分析）

**触发条件：** 用户想了解网络现状、发现问题或给出部署建议——覆盖评估、容量分析、fgOTN/OSU 部署规划、站点选址、企业覆盖缺口、资源瓶颈识别等。报告和大纲只是分析的输出形式，不是触发条件。不适用：与网络分析无关的一般对话、简单知识问答。

**工具调用顺序（有匹配模板）：**
```
search_outline_templates → load_template_outline → [modify_outline]
```

**工具调用顺序（无匹配模板 / 用户拒绝模板）：**
```
search_graph_tree → build_outline_from_anchor → modify_outline
```

**SOP 核心步骤：**

**步骤 1：先找现成模板**
调 `search_outline_templates`，根据 `scene_name`/`summary`/`score` 自行判断是否高度匹配。
- 有匹配 → 调 `load_template_outline` 加载，告知用户模板名称，询问是否使用，**等待用户确认，不得自行决定**
- 无匹配 → 直接进入步骤 2，无需告知用户"未找到模板"

**步骤 2：从知识库实时构建**
调 `search_graph_tree`，按锚节点选择原则选定锚节点：选与用户需求最直接对应的节点，优先 L3/L4 乃至 query 节点，避免选 L1/L2 等过于宽泛的顶层节点。

**步骤 3：展开并主动修剪**
调 `build_outline_from_anchor` 后，**不等用户指示**，立即通过一次 `modify_outline` 完成：
1. 结构修剪：删除与用户需求无关的节点，或用 `keep_only_node` 保留关键分支
2. 范围过滤：若用户在需求中已指定分析范围（城市、行业、时间段、阈值等），用 `modify_node_description` 将过滤条件写入所有相关 query 节点描述

两项均无需操作时可不调用。修改完成后简短告知用户，询问是否进一步调整。

**步骤 4：按用户反馈修改**
调 `modify_outline`，多个独立操作合并为一次调用。`modify_node_description` 修改 query 节点描述时，该描述直接作为数据过滤参数——不得只改 L3/L4 名称而忽略 query 节点描述的同步更新。

**关键约束：**
- 大纲通过 `outline` 事件推送给前端，**禁止在文字回复里输出大纲内容**
- 每次工具调用后文字回复严格控制在 1-2 句话

### 8.2 consolidate-expert（专家知识沉淀）

**触发条件：** 用户发来一段较长的业务描述（通常 80～300 字），内容是自己的分析判断、工作方法或场景经验，而不是在提问。典型表现："我们一般怎么看……"、"这个场景需要关注……"、"根据我的经验……"、直接把一段业务思路一次性发过来。无论用户有没有说"保存"，只要是在输出业务知识就触发。不适用：用疑问句提问、请求生成报告、简短对话。

**工具调用顺序：**
```
search_graph_tree → set_outline_from_markdown → set_scene_metadata → [modify_outline] → save_outline_template
```

**SOP 核心步骤：**

**步骤 1：检索知识库**
调 `search_graph_tree`，将专家描述的业务场景**完整原文**传入 `question`。记录返回的 query 节点 ID，后续构造大纲时 query 节点只能引用这些 ID，不可新建。

**步骤 2：构造大纲**
根据专家输入和知识库节点，自行设计大纲结构，调 `set_outline_from_markdown` 传入 `md_with_ids`。L2/L3/L4 由 LLM 按专家意图自由设计（可新建），query 节点只能引用步骤 1 返回的知识库节点 ID。调用后大纲立即展示给专家。

**步骤 3：填写场景元数据**
`set_outline_from_markdown` 调用完毕后**立即**调 `set_scene_metadata`，填写 `scene_name`/`summary`/`keywords`/`usage_conditions`。

**步骤 4：按专家意见修改（可选）**
调 `modify_outline`，一句话确认变更后询问是否满意。

**步骤 5：保存为模板**
只在专家明确确认时（说"保存"、"就这样"、"好的"等）调 `save_outline_template`，不主动催促。

**支持文件：** `skills/consolidate-expert/node-text-format.md`，详细说明 `md_with_ids` 的格式规则，SOP 中通过 `read_skill("consolidate-expert", "node-text-format.md")` 按需加载（Level 2）。

---

## 9. Memory 设计

实现在 `agent_with_skills/memory.py`，`AgentWithSkillsMemory` 继承自 `memory/store.py` 的 `AgentMemory`。

### 状态字段

#### AgentMemory（基类）

| 字段 | 类型 | 说明 |
|------|------|------|
| `outline_tree` | dict | 当前大纲的 JSON 树，空 dict 表示无大纲 |
| `markdown` | string | 当前大纲的纯 Markdown 视图（前端渲染用） |
| `md_with_ids` | string | 当前大纲的带 ID 视图，每轮注入 system prompt |
| `kb_tree_text` | string | 最近一次 `search_graph_tree` 的返回文本 |
| `_history` | list[dict] | OpenAI 格式对话历史（role/content/tool_calls） |

#### AgentWithSkillsMemory（扩展）

| 字段 | 类型 | 说明 |
|------|------|------|
| `extraction` | dict | 场景元数据，由 `set_scene_metadata` 写入，`save_outline_template` 读取 |

`extraction` 结构：`{scene_name, summary, keywords, usage_conditions}`

### 大纲不存入对话历史

大纲以独立字段保存，而不是作为 tool_result 消息存入 `_history`。原因：
- 每次修改后大纲都会更新，旧版本若在 history 中会误导 LLM
- 大纲 JSON 体积较大（几百到几千 token），不应在每轮对话中重复累积
- LLM 只需要看最新大纲，通过 system prompt 注入一次即可

### build_messages（每轮 LLM 调用前构建）

```python
def build_messages(self, system_prompt: str) -> list[dict]:
    content = system_prompt

    # consolidate-expert 场景：将场景元数据追加到 system prompt
    if self.has_extraction:
        content += f"\n\n## 当前场景元数据\n场景名：{...}\n关键词：{...}\n使用条件：{...}"

    # 有大纲时追加到 system prompt 末尾（不新增 system message）
    if self.has_outline:
        content += f"\n\n## 当前大纲（可通过节点ID引用）\n\n{self.md_with_ids}"

    return [{"role": "system", "content": content}, *self._history]
```

大纲和元数据拼接在 system prompt 内容末尾，而不是独立的 system message，因为大多数模型要求 system message 只出现在对话开头。

### System Prompt 缓存

Skill 的 Level 0 列表（name + description）在 `AgentWithSkills.__init__()` 时构建一次并缓存为 `self._system_prompt`，后续每轮 LLM 调用直接使用。Skills 在运行时不变，无需每轮重建。

### reset()

清空 `_history`、`outline_tree`、`markdown`、`md_with_ids`、`extraction`，以及 `agent._loaded` set（已加载的 skill SOP 记录），还原到初始状态供新一轮对话使用。
