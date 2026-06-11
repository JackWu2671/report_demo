# 系统设计文档

## 目录

1. [专家知识库构建](#1-专家知识库构建)
2. [专家知识库检索](#2-专家知识库检索)
3. [Tools 设计](#3-tools-设计)
4. [大纲表达与修改](#4-大纲表达与修改)
5. [SKILL：一句话生成报告](#5-skill一句话生成报告)
6. [SKILL：专家沉淀大纲](#6-skill专家沉淀大纲)
7. [SKILL：知识发布](#7-skill知识发布)

---

## 1. 专家知识库构建

### 1.1 知识结构

知识库采用五层树形结构，每一层对应一类业务粒度：

| 层级 | 节点类型 | 说明 | ID 前缀 |
|------|---------|------|---------|
| L1 | 场景 | 顶层业务主题，如"政企OTN升级" | `L1_xxx` |
| L2 | 子场景 | L1 下的分析方向，如"fgOTN部署" | `L2_xxx` |
| L3 | 评估维度 | 分析视角，如"传送网络覆盖企业分析" | `L3_xxx` |
| L4 | 评估项 | 具体分析板块，如"站点覆盖企业分析" | `L4_xxx` |
| L5 | 评估指标（query 节点） | 可执行的数据查询，挂载 SQL 和渲染配置 | `L5_xxx` |

L5 是知识库的叶子节点，直接驱动报告中的图表和表格渲染，不能再挂子节点。

### 1.2 节点属性

#### 通用属性（L1–L5 全部具备）

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 节点唯一标识，格式 `L<层级>_<序号>`，如 `L3_016` |
| `level` | int | 层级编号，1–5 |
| `name` | string | 节点名称。**L5 的 name 即查询语句本身**（如"OTN站点价值分布"），系统据此匹配 SQL，是指标的唯一标识 |
| `keywords` | list[string] | 3–6 个检索关键词，**仅用于 FAISS 向量索引**，决定该节点能被哪些查询检索到，不在报告中展示 |
| `description` | string | 节点描述。**L1–L4**：50–100 字的业务说明，解释本节分析什么、如何计算、输出什么；**L5**：即查询参数，直接描述数据过滤范围（如"仅统计南宁市的企业行业分布"），为空时表示全量查询 |
| `condition` | string | 展示条件，格式为"当……时，本节才展示"。节点并非在任何情况下都展示时填写，无条件则留空。报告渲染时由前端判断是否渲染该节点 |
| `condition_queries` | list[string] | 条件判断所依赖的数据查询 ID 列表，配合 `condition` 使用 |
| `summarySuggestion` | string | 章节摘要生成提示，LLM 在为该节点生成总结文字时的参考指引，留空则使用通用提示 |

#### L5 专有属性（仅评估指标节点具备）

| 字段 | 类型 | 说明 |
|------|------|------|
| `exec_sql` | string | 执行 SQL，报告生成时直接提交给数据库。可通过 `edit_node` 在 SQL 基础上微调，但只能改已有字段/枚举，不能引入新字段 |
| `apiName` | string | 数据接口名称，与 `exec_sql` 配合使用 |
| `extracted_table` | list | 预置抽样数据，用于 mock 模式下的预览渲染，正式报告不使用此字段 |
| `renderType` | string | 图表类型，决定该指标以何种形式渲染。常见值：`bar`（柱状图）、`line`（折线图）、`pie`（饼图）、`table`（表格）、`number`（单数值卡片）等 |
| `colX` | string | 图表 X 轴（横轴）对应的数据字段名 |
| `colY` | string | 图表 Y 轴（纵轴）对应的数据字段名 |

> **L5 改名的连锁效应**：通过 `edit_node(field="name")` 修改 L5 节点名称时，系统会自动从知识库中查找同名节点，并将 `exec_sql / renderType / colX / colY / apiName / extracted_table` 全部同步过来，同时更新节点 ID。这是"换指标"的标准操作。

### 1.3 原始数据格式

每一层对应一个 JSON 文件，存放在 `backend/expert_knowledge/` 目录：

```
expert_knowledge/
  场景.json       → L1
  子场景.json     → L2
  评估维度.json   → L3
  评估项.json     → L4
  评估指标.json   → L5
```

L5 原始文件中，SQL 等执行信息被打包在 `answer` 字段（JSON 字符串），`build_knowledge_nodes.py` 构建时会自动解包展开为 `exec_sql / apiName / extracted_table` 等独立字段。

构建后的合并节点文件为 `expert_knowledge/node.json`，关系文件为 `expert_knowledge/relation.json`，运行时由 `loader.py` 加载。

### 1.4 构建流程

构建分三步，全部在 `backend/scripts/` 目录执行：

**第一步：合并节点总表**

```bash
cd backend
python scripts/build_knowledge_nodes.py
```

读取五层 JSON，提取 `id / level / name / description / condition / exec_sql` 等字段，
合并输出为 `expert_knowledge/knowledge_nodes.json`。

**第二步：生成父子关系**

```bash
python scripts/build_knowledge_relations.py
```

通过 UUID 匹配（L1→L4）和 name 匹配（L4→L5）推导父子关系，
输出 `expert_knowledge/knowledge_relations.json`，格式为：

```json
[{"parent": "L2_005", "child": "L3_016", "order": 0}, ...]
```

**第三步：构建向量索引**

```bash
python scripts/build_index.py
```

对每个节点拼接 `name + keywords` 文本，调用 Embedding 服务获取向量，
构建 FAISS 索引并保存：

```
data/faiss.index       → 向量索引
data/faiss_id_map.json → 索引位置 → 节点 ID 映射
```

> **注意**：每次通过知识发布新增节点后，必须重新执行第三步重建索引，新节点才能被检索命中。

---

## 2. 专家知识库检索

### 2.1 检索原理

检索入口：`backend/skills/_lib/search_graph_tree.py`，底层依赖 `backend/skills/_lib/retriever.py`。

流程：

```
用户查询文本
    ↓ Embedding 服务向量化
FAISS 向量检索（Top-K 相似节点）
    ↓ 加载父子关系
为每个命中节点补全祖先路径（L5 → L4 → L3 → L2 → L1）
    ↓
组装成带层级树的结果，按相似度标注命中节点（★）
```

### 2.2 输出格式

检索返回两部分内容：

**匹配节点列表**（Top 命中，带相似度分）：
```
=== 匹配节点 ===

[Q L5_341] 站点部署fgOTN单板的建议  (0.7003)
  路径: 政企OTN升级 > fgOTN部署 > fgOTN演进准备度评估 > 站点fgOTN支持度分析 > 站点部署fgOTN单板的建议
```

**相关知识树**（完整祖先路径，★ 标注命中节点）：
```
=== 相关知识树（★ 为命中节点）===

[L1 L1_003] 政企OTN升级
  [L2 L2_005] fgOTN部署
    [L3 L3_016] fgOTN演进准备度评估
      [L4 L4_014] 站点fgOTN支持度分析
        [Q L5_341] 站点部署fgOTN单板的建议  ★0.7003
```

### 2.3 调用方式

**在 Skill 脚本中通过 bash 调用：**

```bash
python3 $SKILLS_DIR/analyze-network/scripts/search_graph_tree.py "查询描述"
```

**在 Agent 内部直接调用（Python）：**

```python
from skills._lib.search_graph_tree import search_graph_tree
result = await search_graph_tree("低阶交叉资源利用率")
# result["status"] == "success" | "not_found"
# result["tree_text"]  → 格式化文本，给 LLM 阅读
# result["graph_tree"] → 结构化 dict，程序处理用
```

### 2.4 节点 ID 的重要性

检索结果中的 **L5 节点 ID**（如 `L5_341`）是后续构造大纲的关键——大纲中的 query 叶子节点必须引用知识库已有 ID，不能新建。L2/L3/L4 节点 ID 在 `add_node` 操作时用作 `parent_id`。

---

## 3. Tools 设计

Agent 拥有四个工具，分为**系统原生工具**和**通过 bash 执行的脚本工具**两类。

### 3.1 系统原生工具（直接 JSON 调用）

这类工具由 Agent 代码直接处理，参数走 JSON 原生类型，不经过 shell，特殊字符（SQL 中的反引号、`<`、`>`、`%` 等）可以安全传入。

---

#### `set_outline` — 一次性写入完整大纲

**用途**：用户或专家自定义报告结构时，整棵覆盖写入大纲。

**参数**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `outline` | `array` | 节点对象数组，顶层恰好一个 L1 根节点 |

**节点对象结构**：

```json
{
  "id": "new_L1_root",
  "name": "报告总标题",
  "description": "50~100字描述",
  "children": [
    {
      "id": "new_L2_001",
      "name": "章节名",
      "description": "描述",
      "children": [
        { "id": "L5_341", "name": "query节点名称" }
      ]
    }
  ]
}
```

**结构约束**：
- 新建结构节点 ID 必须按 `new_L<层级>_<序号>` 命名（根用 `new_L1_xxx`，往下依次 `new_L2_xxx / new_L3_xxx / new_L4_xxx`）
- L5 节点是叶子，必须引用知识库已有 ID，禁止新建
- 根节点 `name` 必须与用户/专家描述中已给出的标题完全一致，原文照用
- `outline` 参数必须传原生数组，**禁止序列化为字符串传入**

**返回**：成功时回显写入后的 YAML 大纲；失败时返回 `写入失败: <原因>`。

---

#### `edit_node` — 直接修改大纲节点属性

**用途**：对已有大纲中的单个节点做局部字段修改，比 `modify_outline.py` 更适合含特殊字符的 SQL 编辑。

**参数**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `node_id` | `string` | 目标节点 ID，如 `L5_071` |
| `field` | `string` | 要修改的字段名 |
| `value` | any | 新值 |

**`field` 枚举值**：

| field | 说明 |
|-------|------|
| `name` | 节点名称；L5 改名后自动从知识库同步 exec_sql / renderType 等所有关联字段 |
| `description` | 节点描述（L5 节点禁止修改） |
| `condition` | 展示条件 |
| `exec_sql` | 查询 SQL，仅限 L5 节点，必须基于原 SQL 改写 |
| `summarySuggestion` | 章节摘要提示 |
| `renderType` | 图表类型 |
| `colX` / `colY` | 图表轴字段 |
| `condition_queries` | 条件查询列表（传数组） |

---

### 3.2 通过 bash 调用的脚本工具

这类工具以 Python 脚本形式存放在各 skill 目录，LLM 通过 `bash` 工具调用。状态通过 session 文件（`/tmp/report_sessions/{session_id}.json`）在 Agent 内存与脚本之间同步。

> **跨平台规则**：命令写在单行，不使用 `\` 换行续接。

---

#### `search_graph_tree.py` — 检索知识图谱

```bash
python3 $SKILLS_DIR/analyze-network/scripts/search_graph_tree.py "查询词"
```

返回命中节点列表及带祖先路径的知识树，供 LLM 在构造大纲时引用节点 ID。

---

#### `get_node_detail.py` — 查询节点完整信息

```bash
python3 $SKILLS_DIR/analyze-network/scripts/get_node_detail.py L5_341 L5_342
```

返回指定节点的所有字段，包括 `exec_sql`、`renderType`、`description` 等。在修改 SQL 前，必须先通过此工具获取原始 SQL。

---

#### `build_outline_from_anchor.py` — 锚节点展开大纲

```bash
python3 $SKILLS_DIR/analyze-network/scripts/build_outline_from_anchor.py --anchor L2_005 [--context "补充说明"]
```

以指定知识库节点为锚点，将其下完整子树展开为报告大纲，写入 session。用于用户提出明确分析场景时快速生成初始大纲。

---

#### `modify_outline.py` — 修改大纲结构

```bash
python3 $SKILLS_DIR/analyze-network/scripts/modify_outline.py "[{\"op\": \"add_node\", \"node_id\": \"L3_016\", \"parent_id\": \"L2_005\"}]"
```

支持的操作（`op` 字段）：

| op | 说明 |
|----|------|
| `add_node` | 从知识图谱添加节点到指定父节点下，`parent_id` 指定挂载位置，`after_id` 指定插入到哪个兄弟节点之后 |
| `delete_node` | 删除节点及其所有子节点 |
| `keep_only_node` | 保留指定节点，删除同级其他兄弟分支 |
| `modify_node_name` | 修改节点名称；L5 改名后自动从知识库同步关联字段 |
| `modify_node_description` | 修改节点描述（L5 禁止） |
| `modify_node_condition` | 设置节点展示条件 |
| `modify_node_exec_sql` | 修改 L5 节点 SQL（仅限 L5） |

---

#### `set_metadata.py` — 写入场景元数据

```bash
python3 $SKILLS_DIR/consolidate-expert/scripts/set_metadata.py \
  --scene-name "传送网络覆盖分析" \
  --summary "一句话摘要（≤50字）" \
  --keywords "OTN,企业覆盖,fgOTN" \
  --usage-conditions "适用条件（≤80字）"
```

---

#### `save_template.py` — 保存为专家模板

```bash
python3 $SKILLS_DIR/consolidate-expert/scripts/save_template.py
```

将当前大纲和元数据保存为可复用模板，供知识发布流程读取。
成功输出 `{"template_id": "...", "scene_name": "...", "path": "..."}`。

---

#### `list_templates.py` / `show_graph.py` / `graph_manage.py` — 知识发布系列

见 [第 7 节](#7-skill知识发布)。

---

### 3.3 工具选择原则

| 场景 | 推荐工具 |
|------|---------|
| 写入全新大纲 | `set_outline`（原生工具） |
| 修改节点的 exec_sql / name / description | `edit_node`（原生工具，特殊字符安全） |
| 在大纲中增删节点、调整结构 | `modify_outline.py`（bash） |
| 检索知识图谱 | `search_graph_tree.py`（bash） |
| 查询节点当前 SQL | `get_node_detail.py`（bash） |

---

## 4. 大纲表达与修改

### 4.1 三视图

大纲在系统内部始终以 `outline_tree`（Python dict）为唯一数据源，根据使用场景派生出三种视图，互不依赖。

| 视图 | 函数 | 用途 | 特点 |
|------|------|------|------|
| **Markdown** | `to_markdown(tree)` | 用户可读，前端渲染 | `#` 标题层级，含 description 和 condition，无 ID |
| **YAML** | `to_yaml(tree)` | 注入 LLM system prompt | 简洁，省略 SQL/渲染字段，保留 id/name/description/condition |
| **clean JSON** | `to_clean_json(tree)` | 程序存储与报告执行 | 完整字段，去除检索内部字段（keywords/score 等） |

**YAML 视图示例**（LLM 看到的大纲）：

```yaml
- id: new_L1_root
  name: 细颗粒功能板部署方案分析
  description: ...
  children:
    - id: new_L2_001
      name: 商业洞察：价值企业覆盖分析
      children:
        - id: L5_318
          name: 高价值2B企业区域OTN站点覆盖率
```

**虚拟根节点**：所有大纲都包裹在 `id = "__root__"` 的虚拟根节点下，Markdown 和 YAML 渲染时自动跳过该层，`add_node` 时 `parent_id=""` 表示挂在虚拟根下。

### 4.2 大纲修改方式

#### 方式一：`edit_node` 工具（修改字段值）

适用于修改单个节点的属性（name、description、exec_sql 等）。参数走 JSON，不过 shell，可以安全传入含特殊字符的 SQL：

```
edit_node(node_id="L5_320", field="exec_sql", value="SELECT ... WHERE city = '南宁'")
```

**L5 改名的特殊行为**：修改 `name` 字段时，如果新名称在知识库中存在，会自动将 `exec_sql / renderType / colX / colY / apiName / extracted_table` 全部同步为知识库的值，并更新节点 ID。

#### 方式二：`modify_outline.py` 脚本（修改树结构）

适用于增删节点、调整层级结构等操作。ops 参数为 JSON 数组，每项为一个操作：

```bash
# 新增节点
python3 $SKILLS_DIR/analyze-network/scripts/modify_outline.py \
  "[{\"op\": \"add_node\", \"node_id\": \"L3_016\", \"parent_id\": \"L2_005\", \"reason\": \"补充演进评估\"}]"

# 删除节点
python3 ... "[{\"op\": \"delete_node\", \"node_id\": \"new_L3_002\"}]"

# 保留指定节点，删除其他同级分支
python3 ... "[{\"op\": \"keep_only_node\", \"node_id\": \"L2_005\"}]"
```

#### 方式三：`set_outline` 工具（整棵重写）

当大纲需要大幅重构时，直接调用 `set_outline` 传入全新结构，覆盖当前大纲。

### 4.3 大纲在 Agent 中的流转

```
脚本修改 session 文件
    ↓
Agent 读回 session，检测到 outline_tree 变化
    ↓
更新 Agent 内存（memory.set_outline）
    ↓
推送 outline 事件到前端（含 markdown / yaml / tree）
    ↓
LLM 下一轮调用时，outline_yaml 自动注入 system prompt
```

---

## 5. SKILL：一句话生成报告

**Skill 名称**：`analyze-network`

**触发条件**：用户提出任何与传送网络相关的分析、评估、规划或决策需求，包括：超千兆升级评估、智能城域网规划、OTN/fgOTN/OSU 网络分析、传送网覆盖评估、容量与资源瓶颈识别、站点选址、企业覆盖缺口等场景。

### 5.1 整体流程

```
用户描述需求
    ↓ Step 1: 理解需求，提取关键词
检索知识图谱 → search_graph_tree.py
    ↓ Step 2: 识别锚节点（最能代表分析场景的 L2~L4 节点）
build_outline_from_anchor.py --anchor <node_id>
    ↓ Step 3: 大纲生成，推送前端
（用户确认或修改大纲）
    ↓ Step 4~8: 按需修改大纲（add/delete/keep/edit）
用户确认 → 生成报告
```

### 5.2 锚节点选择原则

锚节点是整个流程的关键决策点。选择规则：

- **以相似度分数为主依据**，优先选得分最高的节点
- 优先选有实质 description 的 L3/L4 节点（粒度更精准）
- 若用户需求明确对应某个 L2 子场景，直接用 L2 作锚点（会展开完整子树）
- 多场景需求时，选最能代表核心意图的单一锚节点，不拆分多次调用

### 5.3 大纲修改阶段

大纲生成后，根据用户反馈按以下逻辑修改：

| 用户意图 | 操作 | 说明 |
|---------|------|------|
| 聚焦某个子方向 | `keep_only_node` | 保留目标分支，删除其他 |
| 补充某个分析视角 | `add_node` | 从知识库检索后挂载 |
| 删除不需要的板块 | `delete_node` | 直接删除节点 |
| 修改节点名称/描述 | `edit_node` | 字段级修改 |
| 修改 SQL 逻辑 | `edit_node` field=exec_sql | 必须基于原 SQL 改写 |

### 5.4 报告触发

用户确认大纲后，脚本向 session 文件写入 `"generate_report": true`，Agent 检测到后推送 `start_report` 事件，前端启动独立的报告渲染流程。

---

## 6. SKILL：专家沉淀大纲

**Skill 名称**：`consolidate-expert`

**触发条件**：用户发来一段较长的业务描述（通常 80～300 字），内容是他自己的分析判断、工作方法或场景经验（陈述句，非提问），无论是否说"保存"或"沉淀"。

### 6.1 整体流程

```
专家输入业务方法论描述
    ↓ Step 1: 检索知识库
search_graph_tree.py "专家描述原文"
    ↓ Step 2: 构造大纲
set_outline（原生工具，JSON 数组参数）
    ↓ Step 3: 填写场景元数据
set_metadata.py --scene-name ... --summary ... --keywords ... --usage-conditions ...
    ↓ Step 4: 按专家意见修改（按需）
modify_outline.py
    ↓ Step 5: 专家确认后保存
save_template.py
```

### 6.2 大纲构造规则

专家沉淀的大纲结构完全由 LLM 根据专家输入自行设计，不直接采用知识库的节点名：

- **L1 根节点**：名称必须与专家描述中已给出的标题完全一致，原文照用
- **L2/L3/L4**：按专家意图自由命名，体现专家的分析框架，不得用知识库节点名替代
- **L5**：只能引用 `search_graph_tree` 返回的知识库已有 ID，禁止新建
- 每个 L4 节点下至少挂一个知识库 L5 节点

**典型结构**：

```
new_L1_root  细颗粒功能板部署方案分析        ← 专家自定义标题
  new_L2_001  商业洞察：价值企业覆盖分析      ← 专家的分析框架
    new_L3_001  高价值企业分布分析
      new_L4_001  企业行业分布
        L5_311    企业行业分布，按二级行业统计  ← 知识库已有 ID
        L5_313    企业城市分布
  new_L2_002  网络洞察：站点演进评估
    new_L3_003  站点价值分级分析
      L5_320    OTN站点价值分布
```

### 6.3 场景元数据

大纲写入成功后，立即填写元数据，用于后续模板检索和知识发布：

| 字段 | 要求 |
|------|------|
| `--scene-name` | 中文，≤10 字，概括分析场景 |
| `--summary` | 一句话摘要，≤50 字 |
| `--keywords` | 逗号分隔，3～8 个核心领域关键词 |
| `--usage-conditions` | 适用条件，≤80 字 |

### 6.4 保存时机

只在专家明确确认时才调用 `save_template.py`（说"保存"、"就这样"、"好的"等）。模板保存后可通过 `publish-knowledge` skill 融合回知识图谱。

---

## 7. SKILL：知识发布

**Skill 名称**：`publish-knowledge`

**触发条件**：用户主动说"发布知识"、"更新知识库"、"知识发布"，或点击"发布知识"按钮。由管理员或知识负责人定期触发，**不在专家沉淀过程中自动触发**。

### 7.1 整体流程

```
Step 1: 并行获取当前状态
  ├── list_templates.py --with-outline  → 所有已保存的专家模板
  └── show_graph.py                     → 当前知识图谱完整节点树

Step 2: 分析融合方案（LLM 逐节点判断）
  对每个模板节点，判断是新增/丰富/忽略

Step 3: 向用户展示方案，等待确认

Step 4: 执行融合
  graph_manage.py --template-id <id> --add-nodes "[...]" --enrich-nodes "[...]"

Step 5: 完成，提醒重建索引
  python scripts/build_index.py
```

### 7.2 融合判断规则

| 节点类型 | 情况 | 操作 |
|---------|------|------|
| L2/L3/L4 | 图谱中无语义相近节点，有跨场景通用价值 | `add_node`：新增结构节点 |
| L2/L3/L4 | 图谱中已有语义相近节点，模板提供新业务视角 | `enrich_existing`：追加描述 |
| L2/L3/L4 | 过于场景专属，或与已有节点完全重复 | `ignore` |
| L5 query | 图谱中不存在 | `add_node` level=5 |
| L5 query | 图谱中已存在 | `ignore` |
| L1 | 任何情况 | `ignore`（L1 由人工维护） |

### 7.3 `add_node` 必填字段

| 字段 | 说明 |
|------|------|
| `level` | 2 / 3 / 4 / 5 |
| `name` | 节点名称 |
| `keywords` | 3～6 个检索关键词，决定该节点能被哪些查询检索到 |
| `description` | L2~L4：50～100 字业务描述；**L5：即查询参数**，直接决定数据过滤范围（如"仅统计南宁市的企业行业分布"） |
| `parent_id` | 知识图谱中已有节点的 ID |

可选字段 `condition`：节点展示条件，如"当用户选择了城市维度时，本节才展示"，无条件限制时留空。

### 7.4 发布后操作

图谱写入完成后，**必须重建向量索引**，否则新节点不会出现在检索结果中：

```bash
cd backend
python scripts/build_index.py
```

### 7.5 注意事项

- 融合是**累积操作**，已有节点的 description 只追加，不覆盖
- 每次发布前必须重新拉取最新图谱（`show_graph.py`），不依赖上次结果
- 同一概念被多个模板描述时，合并为一条 `add_node` 或 `enrich_existing`，不重复写入
