from sentence_transformers import CrossEncoder
from .schema import Chunk
import torch


def rerank(
    query: str,
    candidates: list[tuple[Chunk, float]],
    model: CrossEncoder,
    top_k: int = 3,
) -> list[tuple[Chunk, float]]:
    pairs = [(query, chunk.text) for chunk, _ in candidates]
    scores = model.predict(
        pairs,
        convert_to_tensor=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    topk_scores, topk_indices = torch.topk(scores, k=top_k)
    return [
        (candidates[index.item()][0], scores[index].item()) for index in topk_indices
    ]
