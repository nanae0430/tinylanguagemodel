from sentence_transformers import CrossEncoder
from .schema import Chunk
import torch


def rerank(
    query: str,
    candidates: list[tuple[Chunk, float]],
    model: CrossEncoder,
    top_k: int = 3,
) -> list[tuple[Chunk, float]]:
    if not candidates:
        return []
    if top_k <= 0:
        raise ValueError("invalid top_k")

    top_k = min(top_k, len(candidates))

    pairs = [(query, chunk.text) for chunk, _ in candidates]
    scores = model.predict(pairs, convert_to_tensor=True)
    topk_scores, topk_indices = torch.topk(scores, k=top_k)
    return [
        (candidates[index.item()][0], score.item())
        for score, index in zip(topk_scores, topk_indices)
    ]
