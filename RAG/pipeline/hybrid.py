from .schema import Chunk
from collections import defaultdict


def hybrid_search(
    dense_results: list[tuple[Chunk, float]],
    sparse_results: list[tuple[Chunk, float]],
    top_k: int = 5,
    c: int = 60,
) -> list[tuple[Chunk, float]]:
    if top_k <= 0:
        raise ValueError("invalid top_k")
    if c <= 0:
        raise ValueError("invalid c")
    chunk_map = {}
    rrf_score = defaultdict(float)
    for i in range(len(dense_results)):
        rrf_score[dense_results[i][0].chunk_id] += 1 / (c + i + 1)
        chunk_map[dense_results[i][0].chunk_id] = dense_results[i][0]
    for i in range(len(sparse_results)):
        rrf_score[sparse_results[i][0].chunk_id] += 1 / (c + i + 1)
        chunk_map[sparse_results[i][0].chunk_id] = sparse_results[i][0]
    return [
        (chunk_map[chunk_id], value)
        for chunk_id, value in sorted(
            rrf_score.items(), key=lambda item: item[1], reverse=True
        )[:top_k]
    ]
