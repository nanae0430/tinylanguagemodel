from sentence_transformers import SentenceTransformer
from .schema import Chunk
from RAG.rag import tokenize, build_bm25_index, bm25_search
import torch


def dense_search(
    query: str,
    embeddings: torch.Tensor,
    chunks: list[Chunk],
    model: SentenceTransformer,
    top_k: int = 5,
) -> list[tuple[Chunk, float]]:
    if top_k <= 0:
        raise ValueError("invalid top_k")
    top_k = min(top_k, len(chunks))
    if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
        raise ValueError("Embeddings and chunks mismatch")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    query_embedding = model.encode(
        query, convert_to_tensor=True, device=device, normalize_embeddings=True
    )
    embeddings = embeddings.to(device)
    topk_logits, topk_indices = torch.topk(query_embedding @ embeddings.T, k=top_k)
    return [
        (chunks[i.item()], logit.item()) for logit, i in zip(topk_logits, topk_indices)
    ]


def sparse_search(
    query: str,
    chunks: list[Chunk],
    bm25_index: dict,
    top_k: int = 5,
) -> list[tuple[Chunk, float]]:
    if top_k <= 0:
        raise ValueError("invalid top_k")
    if bm25_index["num_docs"] != len(chunks):
        raise ValueError("BM25 index and chunks mismatch")
    top_k = min(top_k, len(chunks))
    bm25_values = bm25_search(tokenize(query), bm25_index)
    return [
        (chunks[i], bm25_values[i])
        for i in sorted(
            range(len(bm25_values)), key=lambda i: bm25_values[i], reverse=True
        )[:top_k]
    ]
