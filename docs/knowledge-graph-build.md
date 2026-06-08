# 知识图谱构建流程：node.json 与 relation.json 的生成

本文档说明 `expert_knowledge/node.json` 和 `relation.json` 是如何一步一步生成的。

---

## 整体依赖图

```
场景/子场景/评估维度.xlsx ──┐
                           ├─[parse_scene_xlsx.py]──────────────┐
评估项.xlsx ───────────────┘                                    │
                                                                ├─[build_knowledge_nodes.py]      → node.json
appSampleQuestion.json ────┐                                    │
                           ├─[merge_sample_questions.py]────────┘
sampleQuestion.json ───────┘                                    │
                                                                └─[build_knowledge_relations.py] → relation.json
```

> **注意**：源文件 `*.xlsx`、`appSampleQuestion.json`、`sampleQuestion.json` 不入 git，需在本地准备。
> `node.json` 和 `relation.json` 是入库的静态快照，运行时由 `skills/_lib/loader.py` 和 `subtree.py` 加载。

---

## 阶段 1 — Excel → 中间层 JSON（L1～L4）

### L1～L3：`parse_scene_xlsx.py`

读取 3 个 Excel 文件，逐行解析 `SCENEKEY` + `CONTENT`（JSON 字符串）两列：

| 输入 | 输出 | 层级 |
|------|------|------|
| `场景.xlsx` | `场景.json` | L1 |
| `子场景.xlsx` | `子场景.json` | L2 |
| `评估维度.xlsx` | `评估维度.json` | L3 |

每条记录输出字段：

| 字段 | 说明 |
|------|------|
| `uuid` | 原始 UUID |
| `id` | 短编号，如 `L1_001` |
| `name` | 节点名称 |
| `level` | 层级编号（1/2/3） |
| `description` | 一句话描述 |
| `dimensions` | 子节点列表，每项含 `uuid`、`name`、`rank` |

### L4：`parse_evaluation_item_xlsx.py`

读取 `评估项.xlsx`，额外解析三个手动填写列：

| 列名 | 说明 |
|------|------|
| `CONDITION` | 整体展示条件表达式，为空时从 `expandLogic` 第一行自动提取 |
| `CONDITION_QUERIES` | 条件相关指标名（逗号或换行分隔） |
| `DESCRIPTION` | 章节导语，为空时回退到 `CONTENT.description` |

输出到 `评估项.json`，相比 L1～L3 多出：

| 字段 | 说明 |
|------|------|
| `condition` | 展示条件表达式 |
| `condition_queries` | 条件相关指标名列表 |
| `summarySuggestion` | LLM 总结指令 |
| `template` | 含 `${}` 占位符的渲染模板（原 `expandLogic`） |
| `dimensions` | 关联的评估指标名列表（字符串，非 UUID） |

---

## 阶段 2 — 问答缓存合并 → 评估指标.json（L5）

**脚本：`merge_sample_questions.py`**

将两份问答缓存合并为叶子节点数据：

```
appSampleQuestion.json (~400 条)  ─┐
                                    ├─ 以 uuid 去重 → 评估指标.json (~922 条, L5)
sampleQuestion.json    (~522 条)  ─┘
```

关键处理：
- `question` 字段重命名为 `name`，`id`(UUID) 重命名为 `uuid`
- 重新生成短编号 `L5_001`、`L5_002`...
- `answer` 字段（JSON 字符串）若缺少 `apiName`，自动补全为 `"NL2SQL"`
- 缺失的 `renderType`、`colX`、`colY` 补 `null`

输出字段：

```json
{
  "id":         "L5_001",
  "level":      5,
  "uuid":       "adf960eb-...",
  "name":       "AEC覆盖用户数",
  "answer":     "{\"apiName\": \"NL2SQL\", \"exec_sql\": \"SELECT ...\", \"extracted_table\": \"[...]\"}",
  "domain":     "接入",
  "renderType": "TABLE",
  "colX":       null,
  "colY":       null
}
```

---

## 阶段 3 — 合并所有层级 → `node.json`

**脚本：`build_knowledge_nodes.py`**

读取 5 个层级的 JSON，通过 `extract_node()` 提取统一字段集，合并为一个数组：

```
场景.json + 子场景.json + 评估维度.json + 评估项.json + 评估指标.json
  ↓
knowledge_nodes.json  →  (shutil.copy)  →  node.json
```

通用字段（L1～L5）：

| 字段 | 说明 |
|------|------|
| `uuid` | 原始 UUID |
| `id` | 短编号（`L1_001` ～ `L5_xxx`） |
| `level` | 层级（1～5） |
| `name` | 节点名称 |
| `description` | 描述；L5 节点此字段为空字符串 |
| `condition` | 展示条件 |
| `condition_queries` | 条件相关指标名列表 |
| `summarySuggestion` | LLM 总结指令 |

**L5 专属字段**（从 `answer` JSON 字符串解包）：

| 字段 | 说明 |
|------|------|
| `renderType` | 渲染类型（`TABLE` / `BAR` / `PIE` 等） |
| `colX` / `colY` | 图表轴列名 |
| `apiName` | 固定为 `"NL2SQL"` |
| `exec_sql` | 查询 SQL |
| `extracted_table` | 涉及的数据表列表 |

---

## 阶段 4 — 推导父子关系 → `relation.json`

**脚本：`build_knowledge_relations.py`**

先从所有层级 JSON 建立两张查找表，再按规则推导关系：

```
uuid_to_id: { uuid → id }   所有层级均建立
name_to_id: { name → id }   仅 L5 节点建立
```

**关系推导规则：**

| 层级对 | 匹配方式 |
|--------|----------|
| L1 → L2 | 父节点 `dimensions[i].uuid` → `uuid_to_id` → 子节点 `id` |
| L2 → L3 | 同上 |
| L3 → L4 | 同上 |
| L4 → L5 | 父节点 `dimensions[i]`（字符串名称）→ `name_to_id` → 子节点 `id` |

输出格式：

```json
{ "parent": "L1_001", "child": "L2_001", "order": 1 }
```

`order` 字段来源：L1～L3 取 `dimensions[i].rank`，L4→L5 取数组下标（从 1 起）。

最终写入：

```
knowledge_relations.json  →  (shutil.copy)  →  relation.json
```

---

## 运行顺序

```bash
# 1. L1～L3：Excel → JSON
python3 backend/scripts/parse_scene_xlsx.py

# 2. L4：Excel → JSON
python3 backend/scripts/parse_evaluation_item_xlsx.py

# 3. L5：合并问答缓存
python3 backend/scripts/merge_sample_questions.py

# 4. 生成 node.json
python3 backend/scripts/build_knowledge_nodes.py

# 5. 生成 relation.json
python3 backend/scripts/build_knowledge_relations.py
```

步骤 4 和 5 可并行执行（均依赖步骤 1～3 的输出，但互不依赖）。
