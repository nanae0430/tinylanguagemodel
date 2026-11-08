from sentence_transformers import SentenceTransformer, CrossEncoder

from RAG.evaluation import RetrievalDataset, metrics
from RAG.pipeline.storage import load_index, load_bm25_index
from RAG.pipeline.query_pipeline import retrieve

import torch
import numpy as np
import time


def measure_time(fn):
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()
    result = fn()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return result, time.perf_counter() - start


data = RetrievalDataset("scifact_benchmark/mini", split="dev")

embeddings, chunks = load_index("RAG/index_scifact")
bm25_index = load_bm25_index("RAG/index_scifact")

embedding_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

rankings = []

for query in data.query_texts:
    results = retrieve(
        query,
        embeddings,
        chunks,
        bm25_index,
        embedding_model,
        reranker_model,
        retrieval_k=50,
        final_k=50,
    )

    # Chunk 排名转 Document 排名，保留首次出现顺序并去重
    ranked_doc_ids = list(dict.fromkeys(chunk.doc_id for chunk, _ in results))
    rankings.append(ranked_doc_ids)

print(metrics(rankings, data.gold_sets))
print("Minimum unique documents:", min(map(len, rankings)))

for i, (ranking, gold) in enumerate(zip(rankings, data.gold_sets)):
    if not (set(ranking[:10]) & gold):
        print("=" * 50)
        print("Query ID:", data.query_ids[i])
        print("Query:", data.query_texts[i])
        print("Gold IDs:", gold)
        print("Retrieved Top10:", ranking[:10])

        for doc_id in gold:
            index = data.doc_id_to_index[doc_id]
            print("Gold Document:", data.doc_texts[index][:500])

from RAG.pipeline.retriever import dense_search, sparse_search
from RAG.pipeline.hybrid import hybrid_search
from RAG.pipeline.reranker import rerank


def check_gold_rank(name, results, gold):
    doc_ids = list(dict.fromkeys(chunk.doc_id for chunk, _ in results))

    doc_rank = next(
        (i for i, doc_id in enumerate(doc_ids, 1) if doc_id in gold),
        None,
    )

    chunk_rank = next(
        (i for i, (chunk, _) in enumerate(results, 1) if chunk.doc_id in gold),
        None,
    )

    print(
        f"{name:15s} " f"Chunk Rank: {str(chunk_rank):5s} " f"Document Rank: {doc_rank}"
    )


for query_id in ["870", "831"]:
    idx = data.query_ids.index(query_id)
    query = data.query_texts[idx]
    gold = data.gold_sets[idx]

    dense = dense_search(query, embeddings, chunks, embedding_model, top_k=50)
    sparse = sparse_search(query, chunks, bm25_index, top_k=50)
    fused = hybrid_search(dense, sparse, top_k=50)
    reranked = rerank(query, fused, reranker_model, top_k=50)

    print("=" * 60)
    print("Query ID:", query_id)
    print("Gold IDs:", gold)

    check_gold_rank("Dense", dense, gold)
    check_gold_rank("BM25", sparse, gold)
    check_gold_rank("RRF", fused, gold)
    check_gold_rank("Cross-Encoder", reranked, gold)

    gold_chunks = [chunk for chunk, _ in fused if chunk.doc_id in gold]

    if gold_chunks:
        print("Gold Chunk in RRF:")
        print(gold_chunks[0].text[:500])
query = data.query_texts[data.query_ids.index("870")]
gold = data.gold_sets[data.query_ids.index("870")]

for k in [50, 100, 200, 500]:
    dense = dense_search(query, embeddings, chunks, embedding_model, top_k=k)
    sparse = sparse_search(query, chunks, bm25_index, top_k=k)

    print(f"\n===== Top{k} =====")
    check_gold_rank("Dense", dense, gold)
    check_gold_rank("BM25", sparse, gold)

for query_id in ["870", "831"]:
    idx = data.query_ids.index(query_id)
    query = data.query_texts[idx]
    gold = data.gold_sets[idx]

    print(f"\n========== Query {query_id} ==========")

    for k in [50, 100, 200]:
        dense = dense_search(query, embeddings, chunks, embedding_model, top_k=k)
        sparse = sparse_search(query, chunks, bm25_index, top_k=k)
        fused = hybrid_search(dense, sparse, top_k=k)
        reranked = rerank(query, fused, reranker_model, top_k=k)

        print(f"\n----- Retrieval K = {k} -----")
        check_gold_rank("Dense", dense, gold)
        check_gold_rank("BM25", sparse, gold)
        check_gold_rank("RRF", fused, gold)
        check_gold_rank("Cross-Encoder", reranked, gold)

stage_rankings = {
    "BM25": [],
    "Dense": [],
    "RRF": [],
    "RRF + CrossEncoder": [],
}


def to_doc_ids(results):
    return list(dict.fromkeys(chunk.doc_id for chunk, _ in results))


for query in data.query_texts:
    dense = dense_search(query, embeddings, chunks, embedding_model, top_k=50)
    sparse = sparse_search(query, chunks, bm25_index, top_k=50)
    fused = hybrid_search(dense, sparse, top_k=50)
    reranked = rerank(query, fused, reranker_model, top_k=50)

    stage_rankings["BM25"].append(to_doc_ids(sparse))
    stage_rankings["Dense"].append(to_doc_ids(dense))
    stage_rankings["RRF"].append(to_doc_ids(fused))
    stage_rankings["RRF + CrossEncoder"].append(to_doc_ids(reranked))


for name, rankings in stage_rankings.items():
    scores = metrics(rankings, data.gold_sets)

    print(
        f"{name:20s} "
        f"Recall@10={scores['Recall@10']:.4f} "
        f"Hit@10={scores['Hit@10']:.4f} "
        f"MRR@10={scores['MRR@10']:.4f}"
    )
warm_up = 10
dense_times = []
sparse_times = []
rrf_times = []
rerank_times = []
for query in data.query_texts[:warm_up]:
    # Dense
    dense, dense_time = measure_time(
        lambda: dense_search(query, embeddings, chunks, embedding_model, top_k=50)
    )
    # BM25
    sparse, sparse_time = measure_time(
        lambda: sparse_search(query, chunks, bm25_index, top_k=50)
    )
    # RRF
    fused, rrf_time = measure_time(lambda: hybrid_search(dense, sparse, top_k=50))
    # Cross-Encoder
    reranked, rerank_time = measure_time(
        lambda: rerank(query, fused, reranker_model, top_k=50)
    )

for query in data.query_texts:
    # Dense
    dense, dense_time = measure_time(
        lambda: dense_search(query, embeddings, chunks, embedding_model, top_k=50)
    )
    # BM25
    sparse, sparse_time = measure_time(
        lambda: sparse_search(query, chunks, bm25_index, top_k=50)
    )
    # RRF
    fused, rrf_time = measure_time(lambda: hybrid_search(dense, sparse, top_k=50))
    # Cross-Encoder
    reranked, rerank_time = measure_time(
        lambda: rerank(query, fused, reranker_model, top_k=50)
    )

    dense_times.append(dense_time)
    sparse_times.append(sparse_time)
    rrf_times.append(rrf_time)
    rerank_times.append(rerank_time)

print(
    f"dense_time:{sum(dense_times)/len(dense_times)}\n"
    f"sparse_times:{sum(sparse_times)/len(sparse_times)}\n"
    f"rrf_times:{sum(rrf_times)/len(rrf_times)}\n"
    f"rerank_times:{sum(rerank_times)/len(rerank_times)}\n"
)


total_times = [
    sum(values) for values in zip(dense_times, sparse_times, rrf_times, rerank_times)
]

for name, times in {
    "Dense": dense_times,
    "BM25": sparse_times,
    "RRF": rrf_times,
    "Reranker": rerank_times,
    "Total": total_times,
}.items():
    print(
        f"{name:12s} "
        f"Mean: {np.mean(times)*1000:.2f} ms  "
        f"P95: {np.percentile(times, 95)*1000:.2f} ms"
    )
