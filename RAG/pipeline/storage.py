import torch, json
from dataclasses import asdict
from .schema import Chunk


def save_index(embeddings: torch.Tensor, chunks: list[Chunk], directory: str):
    assert embeddings.shape[0] == len(chunks)
    with open(directory + r"/index/chunks.json", "w", encoding="utf-8") as f:

        chunk_data = [asdict(chunk) for chunk in chunks]
        json.dump(chunk_data, f, ensure_ascii=False, indent=2)
    torch.save(embeddings.cpu(), directory + r"/index/embeddings.pt")


def load_index(directory: str) -> tuple[torch.Tensor, list[Chunk]]:
    with open(directory + r"/index/chunks.json", "r", encoding="utf-8") as f:
        chunk_data = json.laod(f)
        chunks = [Chunk(**chunk) for chunk in chunk_data]
    embeddings = torch.load(
        directory + r"/index/embeddings.pt", map_location="cpu", weights_only=True
    )

    return (embeddings, chunks)
