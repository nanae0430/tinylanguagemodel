from .schema import Chunk
from sentence_transformers import SentenceTransformer
import torch


def encode_chunks(
    chunks: list[Chunk],
    model: SentenceTransformer,
    batch_size=32,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    if len(chunks) == 0:
        return torch.zeros(
            size=(0, model.get_embedding_dimension()),
            dtype=torch.float32,
            device=device,
        )
    return model.encode(
        [chunk.text for chunk in chunks],
        convert_to_tensor=True,
        device=device,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
