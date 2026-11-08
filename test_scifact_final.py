from sentence_transformers import SentenceTransformer, CrossEncoder

from RAG.evaluation import RetrievalDataset, metrics
from RAG.pipeline.storage import load_index, load_bm25_index
from RAG.pipeline.query_pipeline import retrieve

import torch
import numpy as np
import time

data = RetrievalDataset("scifact_benchmark/mini", split="test")


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

    ranked_doc_ids = list(dict.fromkeys(chunk.doc_id for chunk, _ in results))
    rankings.append(ranked_doc_ids)

scores = metrics(rankings, data.gold_sets)

print("Test Queries:", len(data.query_texts))
print("Recall@10:", scores["Recall@10"])
print("Hit@10:", scores["Hit@10"])
print("MRR@10:", scores["MRR@10"])
