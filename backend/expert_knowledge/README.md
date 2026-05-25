# expert_knowledge 目录说明

本目录存放知识图谱和 query 节点的相关数据文件。

---

## 文件一览

### 1. 知识图谱（已入库，跟随代码提交）

#### `node.json`
知识图谱的**节点定义**，共 14 个节点，分 L1～L5 五个层级。

| 层级 | 含义 | 节点数 |
|------|------|--------|
| L1 | 顶层业务方向（如"政企OTN升级"） | 1 |
| L2 | 二级方向（如"fgOTN部署"、"量子加密板部署"） | 2 |
| L3 | 分析大类（如"传送网络覆盖分析"、"容量分析"） | 2 |
| L4 | 具体分析模块（如"企业分布分析"、"设备槽位资源分析"） | 5 |
| L5 | **Query 节点**，叶子节点，description 直接作为数据查询参数 | 14 |

被 `skills/_lib/loader.py` 加载，用于语义检索和大纲构建。

#### `relation.json`
知识图谱的**父子关系**，格式为 `{parent, child, order}`。
与 `node.json` 配合，由 `skills/_lib/subtree.py` 构建子树，驱动大纲展开。

---

### 2. Query 节点 SQL 样本（不入库，本地使用）

> ⚠️ 以下三个文件因数据量大或含敏感信息，**不提交到 git**，仅在本地使用。

#### `appSampleQuestion.json`（源文件，本地）
来源于 APP 场景的问答缓存，共约 **400 条**。
原始字段较多（含 `scene`、`classification`、`cacheFrom` 等），实际只用其中 7 个字段。

#### `sampleQuestion.json`（源文件，本地）
来源于 dataEval 场景的问答缓存，共约 **522 条**。
字段与 appSampleQuestion 大体相同，但 `renderType`、`colX`、`colY` 可能缺失。

#### `评估指标.json`（**实际使用**，本地）
由 `backend/scripts/merge_sample_questions.py` 将上面两个源文件合并而成，共约 **922 条**。
**代码中应读取此文件**，不要直接读源文件。

字段结构（按顺序）：

```json
{
  "nodeId":     "L5_001",
  "level":      5,
  "id":         "adf960eb-...",
  "name":       "AEC覆盖用户数",
  "answer":     "{\"apiName\": \"NL2SQL\", \"exec_sql\": \"SELECT ...\", \"extracted_table\": \"[...]\"}",
  "domain":     "接入",
  "renderType": "TABLE",
  "colX":       null,
  "colY":       null
}
```

| 字段 | 说明 |
|------|------|
| `nodeId` | 短编号 `L5_001`...，LLM 大纲可见 |
| `level` | 固定为 `5`，对应评估指标层级 |
| `id` | 原始 UUID，合并时用于去重 |
| `name` | 指标名称（自然语言） |
| `answer` | JSON 字符串，含 `apiName`（固定为 `"NL2SQL"`）、`exec_sql`（SQL语句）和 `extracted_table`（涉及的表） |
| `domain` | 业务域，如"接入"、"数通" |
| `renderType` | 渲染类型：`"TABLE"` / `"BAR"` / `"PIE"` 等，可为 `null` |
| `colX` | 图表 X 轴列名，可为 `null` |
| `colY` | 图表 Y 轴列名，可为 `null` |

---

### 3. 参考样例（已入库）

#### `sample_query_sql.json`
从 `mergedSampleQuestions.json` 中挑出的 **2 条典型记录**，用于开发时快速了解数据结构，已提交到 git。
- 第 1 条：有 `renderType: "TABLE"`，`answer` 含 `apiName: "NL2SQL"`
- 第 2 条：`renderType` 为 `null`，展示缺失字段的情况

---

## 合并脚本

```bash
# 将 appSampleQuestion.json 和 sampleQuestion.json 合并为 mergedSampleQuestions.json
python3 backend/scripts/merge_sample_questions.py
```

脚本位于 `backend/scripts/merge_sample_questions.py`，自动去重（以 `id` 为键），缺失字段补 `null`。
