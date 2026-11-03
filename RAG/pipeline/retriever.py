from sentence_transformers import SentenceTransformer
from .schema import Chunk
import torch


def dense_search(
    query: str,
    embeddings: torch.Tensor,
    chunks: list[Chunk],
    model: SentenceTransformer,
    top_k: int = 5,
) -> list[tuple[Chunk, float]]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    query_embedding = model.encode(
        query, convert_to_tensor=True, device=device, normalize_embeddings=True
    )
    embeddings = embeddings.to(device)
    topk_logits, topk_indices = torch.topk(query_embedding @ embeddings.T, k=top_k)
    return [
        (chunks[i.item()], logit.item()) for logit, i in zip(topk_logits, topk_indices)
    ]
