from .schema import Chunk
from collections import defaultdict


def hybrid_search(
    dense_results: list[tuple[Chunk, float]],
    sparse_results: list[tuple[Chunk, float]],
    top_k: int = 5,
    c: int = 60,
) -> list[tuple[Chunk, float]]:
    rrf_score = defaultdict(float)
    for i in range(len(dense_results)):
        rrf_score[dense_results[i][0]] += 1 / (c + i)
    for i in range(len(sparse_results)):
        rrf_score[sparse_results[i][0]] += 1 / (c + i)
    return sorted(rrf_score.items(), key=lambda item: item[1], reverse=True)[:top_k]
