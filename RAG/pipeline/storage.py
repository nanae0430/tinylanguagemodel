import torch, json
from dataclasses import asdict
from .schema import Chunk

from pathlib import Path


def save_index(embeddings: torch.Tensor, chunks: list[Chunk], directory: str):
    if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
        raise ValueError("Embeddings and chunks shape mismatch")
    index_dir = Path(directory)
    index_dir.mkdir(parents=True, exist_ok=True)

    with open(index_dir / "chunks.json", "w", encoding="utf-8") as f:

        chunk_data = [asdict(chunk) for chunk in chunks]
        json.dump(chunk_data, f, ensure_ascii=False, indent=2)
    torch.save(embeddings.cpu(), index_dir / "embeddings.pt")


def save_bm25_index(bm25_index: dict, directory: str) -> None:
    index_dir = Path(directory)
    index_dir.mkdir(parents=True, exist_ok=True)

    with open(index_dir / "bm25.json", "w", encoding="utf-8") as f:
        json.dump(bm25_index, f, ensure_ascii=False, indent=2)


def load_index(directory: str) -> tuple[torch.Tensor, list[Chunk]]:
    index_dir = Path(directory)
    with open(index_dir / "chunks.json", "r", encoding="utf-8") as f:
        chunk_data = json.load(f)
        chunks = [Chunk(**chunk) for chunk in chunk_data]
    embeddings = torch.load(
        index_dir / "embeddings.pt", map_location="cpu", weights_only=True
    )
    if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
        raise ValueError("Embeddings and chunks shape mismatch")
    return (embeddings, chunks)


def load_bm25_index(directory: str) -> dict:
    index_dir = Path(directory)

    with open(index_dir / "bm25.json", "r", encoding="utf-8") as f:
        bm25_index = json.load(f)
        for term, posting in bm25_index["inverted_index"].items():
            bm25_index["inverted_index"][term] = {
                int(doc_id): freq for doc_id, freq in posting.items()
            }

    return bm25_index


def save_source_map(
    source_map: dict[str, str],
    directory: str,
) -> None:
    # TODO: 保存为 sources.json
    map_dir = Path(directory)
    map_dir.mkdir(parents=True, exist_ok=True)

    with open(map_dir / "sources.json", "w", encoding="utf-8") as f:
        json.dump(source_map, f, ensure_ascii=False, indent=2)


def load_source_map(
    directory: str,
) -> dict[str, str]:
    # TODO: 从 sources.json 加载
    map_dir = Path(directory)

    with open(map_dir / "sources.json", "r", encoding="utf-8") as f:
        source_map = json.load(f)
    return source_map
