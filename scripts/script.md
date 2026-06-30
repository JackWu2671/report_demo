# 生成 node.json 和 relation.json 的步骤

本文档记录从原始 Excel 数据出发，一步步生成 `reference/node.json` 和 `reference/relation.json` 的完整流程。

所有命令均在 `backend/` 目录下运行。

---

## 前置条件

安装依赖：

```bash
pip install openpyxl
```

如需生成评估项的 `DESCRIPTION` 列（可选步骤），还需配置 `.env`：

```
LLM_BASE_URL=...
LLM_MODEL_NAME=...
LLM_API_KEY=...
```

---

## 数据流总览

```
场景.xlsx / 子场景.xlsx / 评估维度.xlsx
        ↓ parse_scene_xlsx.py
场景.json / 子场景.json / 评估维度.json

评估项.xlsx
        ↓ (可选) generate_description.py   ← 调 LLM 补全 DESCRIPTION 列
        ↓ parse_evaluation_item_xlsx.py
评估项.json

评估指标.xlsx（已有 JSON，不需要脚本）
→ 评估指标.json

以上所有 JSON ──┐
                ↓ build_knowledge_nodes.py
                → knowledge_nodes.json = node.json

以上所有 JSON ──┐
                ↓ build_knowledge_relations.py
                → knowledge_relations.json = relation.json
```

---

## 第一步：解析场景 / 子场景 / 评估维度 Excel

将三张 xlsx 批量转为 JSON，生成 L1/L2/L3 节点。

**输入**（放在 `reference/`）：
- `场景.xlsx`
- `子场景.xlsx`
- `评估维度.xlsx`

**运行**：

```bash
cd backend
python3 scripts/parse_scene_xlsx.py
```

**输出**（写入 `reference/`）：
- `场景.json`（level=1，id 前缀 L1）
- `子场景.json`（level=2，id 前缀 L2）
- `评估维度.json`（level=3，id 前缀 L3）

Excel 必须有 `SCENEKEY` 和 `CONTENT` 两列，`CONTENT` 存放原始 JSON 字符串。

---

## 第二步（可选）：用 LLM 为评估项批量生成 DESCRIPTION

如果 `评估项.xlsx` 里的 `DESCRIPTION` 列为空，可以用此脚本调用 LLM 自动生成章节导语，写回 xlsx。

```bash
cd backend
python3 scripts/generate_description.py
```

- 已有 `DESCRIPTION` 的行会自动跳过（`SKIP_NONEMPTY=True`）。
- 运行前会自动备份原文件为 `评估项.bak.xlsx`。
- 依赖 `.env` 中配置的 LLM 环境变量。

此步骤**可选**，跳过也不影响后续流程，只是 `评估项.json` 里 `description` 字段会留空。

---

## 第三步：解析评估项 Excel

将 `评估项.xlsx` 转为 `评估项.json`，生成 L4 节点。

**输入**（放在 `reference/`）：
- `评估项.xlsx`（列：`SCENEKEY`、`CONTENT`、可选 `CONDITION`、`CONDITION_QUERIES`、`DESCRIPTION`）

**运行**：

```bash
cd backend
python3 scripts/parse_evaluation_item_xlsx.py
```

**输出**：
- `reference/评估项.json`（level=4，id 前缀 L4）

---

## 第四步：确认评估指标 JSON 已就位

`评估指标.json`（level=5，id 前缀 L5）不需要脚本转换，直接放在 `reference/` 即可。

---

## 第五步：生成 node.json

汇总所有层级 JSON，合并为统一节点总表。

**运行**：

```bash
cd backend
python3 scripts/build_knowledge_nodes.py
```

**输出**：
- `reference/knowledge_nodes.json`
- `reference/node.json`（同内容，副本）

脚本按顺序读取 `场景.json / 子场景.json / 评估维度.json / 评估项.json / 评估指标.json`，缺少的文件会跳过并打印提示。

---

## 第六步：生成 relation.json

从各层级 JSON 的 `dimensions` 字段推导父子关系。

**运行**：

```bash
cd backend
python3 scripts/build_knowledge_relations.py
```

**输出**：
- `reference/knowledge_relations.json`
- `reference/relation.json`（同内容，副本）

关系推导规则：
- L1→L2、L2→L3、L3→L4：通过 `dimensions[i].uuid` 匹配子节点的 `uuid`
- L4→L5：通过 `dimensions[i]`（字符串名称）匹配评估指标的 `name`

---

## 完整命令汇总

```bash
cd backend

# 1. 解析场景类 Excel（L1/L2/L3）
python3 scripts/parse_scene_xlsx.py

# 2. 解析评估项 Excel（L4）（需要先完成步骤 1）
python3 scripts/parse_evaluation_item_xlsx.py

# 3. 生成 node.json（需要 L1~L5 的 JSON 全部就位）
python3 scripts/build_knowledge_nodes.py

# 4. 生成 relation.json（需要 L1~L5 的 JSON 全部就位）
python3 scripts/build_knowledge_relations.py
```

如果还需要构建向量索引（用于语义检索），在生成 `node.json` 之后执行：

```bash
python3 scripts/build_index.py
```

输出 `reference/faiss.index` 和 `reference/faiss_id_map.json`。
