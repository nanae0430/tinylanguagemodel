# RAG Retrieval Benchmarks

从零实现并评估稀疏检索、稠密检索、排名融合及 Cross-Encoder 重排序。本目录是 **RAG 检索阶段的实验项目**，重点在算法理解、可复现的检索评测与错误案例分析；**尚未包含 LLM 答案生成、API 服务或生产部署**。

> 当前最佳实验结果：在自建的 SciFact Mini Dev（1,000 篇文档、40 条 Query）上，BM25 Top-10 + Dense Top-10 候选去重后，使用 Cross-Encoder 重排序，得到 **Recall@10 = 0.9750、Hit@10 = 0.9750、MRR@10 = 0.8037**。这是小规模、经简化的开发集实验，**不能直接与官方 BEIR SciFact 榜单比较**。

## 检索流水线

```text
                         Query
                           |
             +-------------+--------------+
             |                            |
     Regex tokenizer               SentenceTransformer
             |                            |
    Indexed BM25 Top-10               Dense Top-10
             |                            |
             +-------------+--------------+
                           |
                  候选并集（去重）
                     最多 20 篇
                           |
                  Cross-Encoder 打分
                           |
                   重排序 Top-10
                           |
              doc_index -> doc_id -> metrics
```

另设 **RRF（Reciprocal Rank Fusion）** 作为独立的融合基线。RRF 使用名次而非原始 BM25 / cosine 分数融合；当前 Cross-Encoder 实验使用的是 **BM25 + Dense 候选并集**，不是 RRF 截断后的 Top-10。

## 已实现

| 模块 | 实现及作用 |
| --- | --- |
| 文本预处理 | 自定义正则分词：小写化、提取字母数字及连字符词语 |
| BM25 | 手写 BM25、TF/IDF、文档长度归一化及倒排索引（postings 保存 term frequency） |
| 索引预计算 | 缓存 postings、doc length、avgdl、IDF，检索时只遍历命中词的 postings |
| Dense | `SentenceTransformer` 编码文档与 Query，向量归一化后进行 cosine Top-K |
| Hybrid | 手写 RRF，利用 `torch.scatter_add_` 按两路检索名次融合 |
| Reranker | `CrossEncoder.predict` 批量处理 Query–Document Pair，将分数映射回候选文档并重排 |
| 评测 | 支持多篇正例的 Recall@K、Hit@K、MRR@K，保留原始文档 ID，统计未命中 Query |

### 文件结构

| 文件 | 作用 |
| --- | --- |
| `RAG.py` | BM25、倒排索引、cosine similarity、RRF 等实现及早期实验代码 |
| `benchimark_scifact.py` | SciFact Mini Dev：BM25、Dense、RRF、Cross-Encoder 对照和失败案例分析（文件名沿用仓库现状） |
| `build_dataset.py` | 从官方 BEIR SciFact ZIP 构建固定种子的 Mini / Full 实验数据 |
| `evaluation.py` | 数据加载、doc_index / doc_id 映射、多正例检索指标 |
| `retrieval_eval.py` | 较早期的小规模检索示例 |
| `test_kit.py` | 辅助实验代码 |

## 实验设置

| 项目 | 设置 |
| --- | --- |
| 来源 | [BEIR SciFact](https://github.com/beir-cellar/beir)，文章标题 + 摘要 |
| 子集 | Mini：1,000 Documents、200 Queries，其中 40 Dev、160 Test |
| 数据构建 | `build_dataset.py`，默认 `seed=39`；保证选中 Query 的正例文档包含在 Mini 语料中 |
| 评测分组 | `split="dev"`，40 Queries；指标支持一条 Query 对应多篇 Gold 文档 |
| BM25 | `k1=1.5`，`b=0.75`，预先构建倒排索引 |
| Dense 模型 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| RRF | BM25 Top-10 + Dense Top-10，`c=60`，最终 Top-10 |
| Cross-Encoder | `cross-encoder/ms-marco-MiniLM-L6-v2`，`batch_size=32`，对两路 Top-10 的去重并集重排序 |
| 评测指标 | Recall@1/3/5/10、Hit@1/3/5/10、MRR@1/3/5/10 |

**数据口径说明：** Mini 是从官方 SciFact 数据中采样后的自定义小语料，Dev/Test 是在原始标注过的 Query 中按本项目方案重新划分。缩小语料及确保正例入选都会影响检索难度，结果不应被描述成官方 SciFact 全量测试集的成绩。

## SciFact Mini Dev 实验结果

数值来自本地实验输出，保留四位小数（40 条 Query），不是跨随机种子均值。

| Metric | BM25 | Dense | RRF | Cross-Encoder |
| --- | ---: | ---: | ---: | ---: |
| Recall@1 | 0.6271 | 0.4896 | 0.5896 | **0.6646** |
| Recall@3 | 0.7854 | 0.6979 | 0.7729 | **0.8292** |
| Recall@5 | 0.8354 | 0.7604 | 0.8229 | **0.8854** |
| Recall@10 | 0.8875 | 0.7917 | 0.9417 | **0.9750** |
| Hit@10 | 0.9000 | 0.8000 | 0.9500 | **0.9750** |
| MRR@10 | 0.7763 | 0.6349 | 0.7392 | **0.8037** |

### 结果分析

- **BM25 优于所选通用 Dense 基线**：在这批科学文献 Query 上，词汇精确匹配表现更强；不能据此推断所有 Dense 模型都较差。
- **两路检索具有互补性**：BM25 单独命中 36/40，Dense 单独命中 32/40；各取 Top-10 后的候选并集覆盖 **39/40（97.5%）**。这衡量的是候选覆盖，不等于融合后的最终 Hit@10。
- **RRF 提升较大 K 下的召回，但未改善早期排序**：相比 BM25，RRF 的 Recall@10 从 0.8875 提升到 0.9417，MRR@10 则从 0.7763 降为 0.7392。
- **Cross-Encoder 重排序改善了最终排序**：在 BM25 + Dense 的去重候选集合上重新评分，最终 Recall@10、Hit@10 均为 0.9750，MRR@10 达到 0.8037。
- **对照范围限制**：RRF 的输入是两路 Top-10、输出最终 Top-10；Cross-Encoder 的输入是两路 Top-10 的 **并集（最多 20 篇）**，二者候选集合大小不同。因此不能把所有差异都归因为重排序模型本身。

### 失败案例

| Query index | Query 摘要 | BM25 | Dense | RRF | Cross-Encoder |
| --- | --- | --- | --- | --- | --- |
| 1 | Safe places / homelessness | Top-10 未命中 | 命中 | 命中 | 命中 |
| 2 | Obesity / life quality | 未命中 | 未命中 | 未命中 | **未命中** |
| 5 | IFIT1 / viral replication | 未命中 | 命中 | 命中 | 命中 |
| 9 | p16INK4A / OPMLs | 未命中 | 命中 | 未命中 | 命中 |

最终 Cross-Encoder 的未命中 Query 索引为 `[2]`。其 Gold 文档不在当前 BM25、Dense 各自 Top-10 的候选集合中，重排序无法凭空补回未召回的文档。

## 运行与复现

以 Python 3.10+ 为例，需要 PyTorch 和 `sentence-transformers`。推荐使用虚拟环境；PyTorch 请按实际 CUDA/CPU 平台安装。以下命令从 **仓库根目录**进入 RAG 目录后执行：

```bash
git clone https://github.com/nanae0430/tinylanguagemodel.git
cd tinylanguagemodel/RAG

python -m pip install torch sentence-transformers

# 首次生成数据（默认构建 1,000 docs / 200 queries、40 dev）
python build_dataset.py --no-full

# 运行 BM25 / Dense / RRF / Cross-Encoder 对照实验
python benchimark_scifact.py
```

`build_dataset.py` 会下载官方 SciFact 数据包；也可以提前下载后用 `--source PATH` 指定本地 ZIP。数据输出到 `RAG/scifact_benchmark/mini/`，其中包含 `corpus.jsonl`、`queries.jsonl` 和 `qrels/`。语料和模型权重不随 Git 仓库提交；模型在首次调用时由 Hugging Face 下载，网络环境与缓存位置会影响首次运行时间。

**当前兼容性说明：** `benchimark_scifact.py` 使用 `from rag import ...`，而实际文件名为 `RAG.py`。这在本项目使用的 Windows 环境中可以运行，但在区分大小写的 Linux/macOS 文件系统上可能报模块找不到；需将导入改为 `from RAG import ...`（在此目录执行脚本）后再运行。该脚本尚未封装为跨平台 CLI。

## 评测口径与注意事项

- **Recall@K**：每条 Query 在 Top-K 中检索到的相关文档数 / 其所有相关文档数，再对 Query 求平均。
- **Hit@K**：至少找到一篇相关文档的 Query 比例。
- **MRR@K**：每条 Query 第一篇相关文档的排名倒数（未命中取 0），再求平均。
- `doc_index` 是文档在当前语料列表中的位置；`doc_id` 是原始数据中的唯一 ID。检索 / Reranker 操作 index，`metrics()` 使用 ID 对齐 qrels。
- BM25 的索引构建与查询计时在当前脚本中分开；查询段包含 Python 评分结果转 Tensor 的时间，不代表完整端到端线上延迟。Cross-Encoder 耗时当前未单独纳入统一对比。
- Dense 的文档 Embedding 在每次 Benchmark 运行时重新生成，尚未做持久化或 ANN 向量索引。Cross-Encoder 每次对候选 Query–Document Pair 联合编码。
- 当前模型是通用文本检索基线，长文本截断、领域适配及重复实验仍有改进空间。

## 后续工程计划

- [x] BM25 / 倒排索引、Dense、RRF 和 Cross-Encoder 对照实验
- [x] 多正例检索评测与失败案例分析
- [ ] 在独立 Test split 与不同随机种子下复核结论
- [ ] 文档加载、Chunking、索引与 Embedding 持久化
- [ ] 构建检索上下文、调用 LLM 生成答案并返回来源引用
- [ ] FastAPI 接口、向量数据库及 Docker 部署
- [ ] Agent 工具调用、状态管理、失败恢复与日志追踪

本目录目前是 **retrieval benchmark / prototype**，不是可直接上线的完整 RAG 服务。