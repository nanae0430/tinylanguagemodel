from sentence_transformers import SentenceTransformer, CrossEncoder
from .schema import Chunk
from .retriever import dense_search, sparse_search
from .hybrid import hybrid_search
from .reranker import rerank
import torch


def retrieve(
    query: str,
    embeddings: torch.Tensor,
    chunks: list[Chunk],
    bm25_index: dict,
    embedding_model: SentenceTransformer,
    reranker_model: CrossEncoder,
    retrieval_k: int = 10,
    final_k: int = 3,
) -> list[tuple[Chunk, float]]:

    dense_results = dense_search(
        query, embeddings, chunks, embedding_model, retrieval_k
    )
    sparse_results = sparse_search(query, chunks, bm25_index, retrieval_k)
    hybrid_results = hybrid_search(dense_results, sparse_results, retrieval_k)
    return rerank(query, hybrid_results, reranker_model, final_k)
