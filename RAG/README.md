# RAG Pipeline：从检索算法到可评测的问答原型

本目录记录一个基于 Python / PyTorch 的 RAG（Retrieval-Augmented Generation）学习与工程实践项目：先手写 BM25 和 RRF 等检索核心逻辑，再组合 SentenceTransformer、Cross-Encoder 与 DeepSeek API，构建可持久化、可测试的完整检索增强生成流程。

**当前进度（2026-11）：** 已完成 TXT/Markdown/JSONL 加载、字符级 Chunking、离线索引、Hybrid Retrieval、Cross-Encoder 重排序、Prompt 构建、LLM 生成、来源追踪和基础引用编号校验；已在 SciFact Mini 上完成独立 Dev/Test 检索评测。**尚未完成真实技术文档上的生成质量评测，也尚未提供 FastAPI、Docker 或生产级服务。**

## 1. 系统架构

~~~text
                          原始文档（TXT / MD / SciFact JSONL）
                                       |
                                Document Loader
                                       |
                             Chunker（500 字符 / 100 overlap）
                                       |
                       +---------------+----------------+
                       |               |                |
                Dense Embeddings    BM25 Index        Source Map
                       |               |                |
                       +------ 离线保存 / 加载 ----------+
                                       |
                                      Query
                                       |
                        +--------------+--------------+
                        |                             |
                 Dense Retriever                BM25 Retriever
                  Top-K Chunks                   Top-K Chunks
                        |                             |
                        +-------------+---------------+
                                      |
                             RRF（Chunk 级融合）
                                      |
                               Cross-Encoder
                                      |
                              Top-N Retrieved Chunks
                                      |
                              Prompt Builder
                                      |
                        DeepSeek（OpenAI-compatible API）
                                      |
                           Answer + [n] Citations
                                      |
                       Citation Number Check + Source Map
~~~

- **BM25**：手写倒排索引、TF/IDF、文档长度归一化和查询评分。
- **Dense**：SentenceTransformer 编码，归一化向量内积检索。
- **RRF**：按两路候选的名次融合，使用 `chunk_id` 合并相同 Chunk。
- **Reranker**：Cross-Encoder 对 Query–Chunk Pair 重新打分。
- **Generation**：将检索上下文拼接为 Prompt，调用 DeepSeek 生成带 `[1]`、`[2]` 等引用的答案。
- **Traceability**：保留 `doc_id -> source`，展示原始文档或 JSONL 记录来源；引用校验目前仅检查编号是否有效。

## 2. 目录与主要文件

以下路径均从仓库根目录 `tinylanguagemodel/` 计：

| 路径 | 功能 |
| --- | --- |
| `RAG/rag.py` | BM25、tokenize、旧版 RRF 与检索算法实验 |
| `RAG/pipeline/schema.py` | `Document` / `Chunk` 数据结构 |
| `RAG/pipeline/loader.py` | 递归读取 TXT/MD，或读取 SciFact JSONL |
| `RAG/pipeline/chunker.py` | 字符级切分与重叠窗口 |
| `RAG/pipeline/embedding.py` | 批量生成归一化 Chunk Embedding |
| `RAG/pipeline/index_builder.py` | 构建和保存 Dense、BM25、来源映射索引 |
| `RAG/pipeline/storage.py` | 索引的 JSON / PyTorch 保存和加载 |
| `RAG/pipeline/retriever.py` | Dense / BM25 检索 |
| `RAG/pipeline/hybrid.py` | Chunk 级 RRF |
| `RAG/pipeline/reranker.py` | Cross-Encoder 重排序 |
| `RAG/pipeline/query_pipeline.py` | 串行查询流程 `retrieve()` |
| `RAG/pipeline/prompt_builder.py` | 上下文编号和回答约束 |
| `RAG/pipeline/generator.py` | DeepSeek-compatible Chat Completions |
| `RAG/pipeline/citation.py` | 生成引用 `[n]` 的合法编号校验 |
| `RAG/build_dataset.py` | 构建可复现的 SciFact Mini/Full 数据 |
| `RAG/evaluation.py` | SciFact Qrels 和 Recall/Hit/MRR 计算 |
| `RAG/benchimark_scifact.py` | **历史版本**的整篇 Document 检索对照 |
| `test_pipeline.py` | 各模块功能与 Smoke Test |
| `test_api.py` / `test_rag.py` | LLM API、三文档端到端生成测试 |
| `test_scifact_index.py` | 1,000 文档的离线索引构建检查 |
| `test_scifact_retrieval.py` | Dev 集消融、错误分析、K 对照、延迟 |
| `test_scifact_final.py` | 独立 Test 集的最终检索评测 |

### 数据及索引格式

`Document(doc_id, source, text)` 转为 `Chunk(chunk_id, doc_id, start_char, end_char, text)`。TXT/MD 以相对路径哈希作为 Document ID；SciFact JSONL 则直接保留 `_id`，用于对齐 Qrels。

~~~text
RAG/index_scifact/
├── chunks.json        # Chunk 元数据和文本
├── embeddings.pt      # [num_chunks, embedding_dim] 张量
├── bm25.json          # BM25 postings、IDF、文档长度
└── sources.json       # doc_id -> source
~~~

`*.pt`、`*.jsonl` 和本地 `scifact_benchmark` 数据由仓库的 `.gitignore` 排除，因此仅克隆仓库不代表已下载全部索引或评测数据。请按下述步骤重建。旧版 `RAG/index_scifact/` 中即使有 JSON 文件，也需要与本地重新生成的 `embeddings.pt` 保持一致。

## 3. 快速运行

所有命令都在**仓库根目录**执行；推荐 Python 3.10+，并使用支持 PyTorch 的独立环境。

~~~bash
git clone https://github.com/nanae0430/tinylanguagemodel.git
cd tinylanguagemodel

# torch 请先按本机 CPU / CUDA 平台安装匹配版本
python -m pip install sentence-transformers openai python-dotenv numpy
~~~

### 3.1 生成 SciFact Mini 数据集

仓库根目录已保存 `scifact.zip`；亦可使用 `build_dataset.py` 的自动下载功能。

~~~bash
python RAG/build_dataset.py --source scifact.zip --output scifact_benchmark --no-full
~~~

默认生成：

- `scifact_benchmark/mini/corpus.jsonl`：1,000 篇标题 + 摘要。
- `scifact_benchmark/mini/queries.jsonl`：200 条有标注的 Claim。
- `scifact_benchmark/mini/qrels/dev.tsv`：40 条 Dev Query 的相关文档标签。
- `scifact_benchmark/mini/qrels/test.tsv`：160 条 Test Query 的相关文档标签。

数据生成固定随机种子 `39`，并确保被选中 Query 的已标注正例存在于这 1,000 篇文档中。**这是缩减语料的自定义 Mini Benchmark，不是官方 BEIR SciFact 全量排行榜结果。**

### 3.2 功能测试与离线索引

~~~bash
python test_pipeline.py
python test_scifact_index.py
~~~

`test_scifact_index.py` 使用 `chunk_size=500`、`overlap=100` 构建索引，数据来源为 `scifact_benchmark/mini/corpus.jsonl`，输出至 `RAG/index_scifact/`。

一次本地运行记录：

| 指标 | 结果 |
| --- | ---: |
| 文档数量 | 1,000 |
| Chunk 数量 | 4,089 |
| Embedding 形状 | `[4089, 384]` |
| 索引构建总耗时 | 3.75 s |

这是单次实验耗时，不包含后续在线 LLM 生成；与硬件、缓存和模型加载状态有关。

### 3.3 检索评测

~~~bash
# 开发集：四阶段消融、未命中分析、K 对照、阶段延迟
python test_scifact_retrieval.py

# 独立测试集：仅评测固定方案
python test_scifact_final.py
~~~

当前 Chunk 级 Pipeline 的评测配置：

| 项目 | 设置 |
| --- | --- |
| 语料 | SciFact Mini：1,000 Documents / 4,089 Chunks |
| Chunking | 500 字符；overlap 100 字符 |
| Dense Model | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| BM25 | `k1=1.5`，`b=0.75`；英文正则分词 |
| RRF | `c=60`，按 `chunk_id` 计分 |
| Cross-Encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| 检索/融合候选 | 每路 Top50，RRF Top50 |
| 评测重排序输出 | Top50 Chunks，按 `doc_id` 首次出现去重成文档排名 |
| Metrics | Recall@K、Hit@K、MRR@K（文档级 Qrels） |

**重要：** 真实问答流程中 `retrieve()` 的默认参数是 `retrieval_k=10`、`final_k=3`；这里为了计算 Recall@10，评测脚本显式使用 `retrieval_k=50`、`final_k=50`。两者不是相同的线上配置。

### 3.4 三文档端到端 RAG Smoke Test

先运行 `python test_pipeline.py` 以构建 `RAG/index_smoke`。在根目录创建仅供本地使用的 `.env`：

~~~dotenv
DEEPSEEK_API_KEY=your_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=your_available_model
~~~

随后：

~~~bash
python test_api.py
python test_rag.py
~~~

`test_rag.py` 读取本地索引，执行 Dense + BM25 → RRF → Cross-Encoder → Prompt → DeepSeek，输出答案、各引用对应的 Source / Chunk ID / 分数，以及 `Citation valid`。`.env` 已加入 Git 忽略规则，**不要提交真实 API Key**。模型名请使用你的账号实际可用的 DeepSeek 模型。

这个 Smoke Test 已跑通基础生成与引用链路；**尚未在完整 SciFact 或仿真技术语料上完成大规模生成质量评测**。

## 4. SciFact Mini 实验结果

### 4.1 Dev 集四阶段消融（40 Queries）

| 方案 | Recall@10 | Hit@10 | MRR@10 |
| --- | ---: | ---: | ---: |
| BM25 | 0.8417 | 0.8750 | 0.7348 |
| Dense | 0.8917 | 0.9000 | 0.7519 |
| BM25 + Dense → RRF | 0.9354 | 0.9500 | 0.7790 |
| **RRF → Cross-Encoder** | **0.9500** | **0.9500** | **0.8611** |

RRF 比单路检索提高了前十名的召回；Cross-Encoder 进一步提高相关文档的早期排名（MRR@10 从 `0.7790` 升至 `0.8611`）。这是在固定 40 条开发查询上的单次结果，而非跨随机种子统计。

### 4.2 独立 Test 集（160 Queries）

模型和参数固定，不根据 Test 结果继续调参：

| 指标 | Dev（40 Queries） | Test（160 Queries） |
| --- | ---: | ---: |
| Recall@10 | 0.9500 | **0.8956** |
| Hit@10 | 0.9500 | **0.9000** |
| MRR@10 | 0.8611 | **0.7979** |

Test Hit@10 = 0.90，意味着 160 条查询中有 144 条 Top10 至少命中一篇相关文档、16 条未命中。Dev/Test 差异可能受到样本量和查询难度影响，不能直接等同于模型过拟合证据。

### 4.3 检索延迟（Dev，40 Queries）

10 次预热，CUDA 同步，`time.perf_counter()` 计时。下面是分阶段测试的本地实测平均值与 P95；不包括模型加载、API 网络调用和 LLM 生成。

| 阶段 | Mean | P95 |
| --- | ---: | ---: |
| Dense | 6.36 ms | 6.87 ms |
| BM25 | 2.01 ms | 4.06 ms |
| RRF | 0.05 ms | 0.06 ms |
| Cross-Encoder | 34.12 ms | 41.38 ms |
| **Total** | **42.54 ms** | **52.07 ms** |

Cross-Encoder 约占平均检索耗时的 80%，是当前主要延迟来源，但也是 MRR 提升的重要来源。总耗时是四个串行阶段的耗时相加；40 条 Query 的 P95 仅供当前实验参考，不是生产服务的 SLA。

### 4.4 失败案例与 K 实验

**Query 870 — “Obesity decreases life quality.”**

- Gold Document ID：`195689316`。
- Dense Top50、BM25 Top50 均未命中；Dense Top100 找到相关 Chunk（Chunk Rank 54，去重后 Document Rank 32）。
- K=200 时 Gold Chunk 进入 RRF（Document Rank 59），重排后仍为 Document Rank 49，未进入 Top10。
- 因此不能单靠扩大候选 K 解决此类低排名问题。

**Query 831 — “NF2 (Merlin) prevents phosphorylation and subsequent cytoplasmic sequestration of YAP in Drosophila.”**

- Gold Document ID：`1897324`。
- K=50 时 BM25 Document Rank 7，RRF 后 Rank 15，Cross-Encoder 后 Rank 12。
- K=100 / 200 时最终重排序文档排名分别为 20 / 29；扩大 K 不一定改善 Top10。
- 该案例反映 Chunk 级融合可能削弱 BM25 在某些查询上的优势，也暴露了字符级切分导致的上下文不完整问题。

## 5. 历史 Document-Level 检索基线（与当前实验分开）

早期 `RAG/benchimark_scifact.py` 使用**整篇文档**做 BM25 / Dense / RRF 实验，并将 BM25 Top10 与 Dense Top10 的候选并集送给 Cross-Encoder 重排。早期 Dev 记录为：

| 历史方案 | Recall@10 | Hit@10 | MRR@10 |
| --- | ---: | ---: | ---: |
| BM25 | 0.8875 | 0.9000 | 0.7763 |
| Dense | 0.7917 | 0.8000 | 0.6349 |
| RRF | 0.9417 | 0.9500 | 0.7392 |
| 候选并集 → Cross-Encoder | **0.9750** | **0.9750** | **0.8037** |

**不能将历史 0.9750 与当前 0.9500 简单解释为新版退步：** 历史方案按 Document 评测且重排序输入为两路候选并集；当前方案先按 Chunk 检索、RRF 截断，再按 Document 去重，两者输入、模型和候选策略并不相同。历史脚本 `benchimark_scifact.py` 的导入/工作目录兼容性仍需要独立整理。

## 6. 评测口径与已知限制

- **Recall@K**：每条 Query 的相关文档命中比例，按 Query 宏平均；**Hit@K**：至少命中一篇相关文档的查询比例；**MRR@K**：首篇相关文档排名倒数的平均值。
- 现有 `evaluation.py` 使用文档 ID 与 Qrels 匹配；Chunk 检索结果先按 `doc_id` 首次出现去重，后计算 Document 级指标。
- SciFact Mini 只含 1,000 篇采样后的标题与摘要，并确保评测查询的正例文档入选，因此**不等同于官方完整 SciFact 结果**。
- Chunking 依据字符位置切分，可能截断单词/句子，且后续 Chunk 不保证含原标题；Embedding / Cross-Encoder 还有各自的 Token 长度限制。
- 当前 BM25 使用英文正则 tokenizer，不适合直接评估中文专业技术语料；Cross-Encoder 也主要用于英文检索。
- Dense 检索对全部向量做矩阵乘法，尚未引入 FAISS / ANN / 向量数据库；扩大到百万级数据需重新评估内存和延迟。
- Source Tracking 可找回原始文件/JSONL 记录；`validate_citations()` 只检查引用编号是否在有效范围内，**不验证引用语义真实性、事实支持性或拒答质量**。
- API 调用需要网络及密钥；推理延迟统计没有包含 DeepSeek 请求耗时、服务部署和并发压力。
- 暂未提供依赖锁文件、自动化 CI、统一配置管理、日志/Trace 或 FastAPI 服务。

## 7. 下一阶段：仿真技术文档问答评估

计划引入脱敏或可公开使用的仿真技术文档和人工整理的问答样例，验证真正的端到端 RAG 质量。建议记录每条样例的 `question`、`reference_answer`、`source_doc_id` / 证据位置，以及是否应当拒答。后续重点是：

- 检索是否找到正确证据，以及引用是否真实支持答案。
- 多 Chunk / 多文档综合回答、术语准确性和证据不足时的拒答。
- 输出可复现的生成质量评测记录，避免只依赖一次主观观察。
- 之后再将稳定的 RAG Pipeline 封装为 FastAPI 服务、Docker 镜像，并延伸到 Agent Harness 的工具调用、状态管理和 Trace。

**现阶段定位：已完成离线检索评测 + 小型端到端生成 Smoke Test 的 RAG 原型，不是可直接部署的生产系统。**
