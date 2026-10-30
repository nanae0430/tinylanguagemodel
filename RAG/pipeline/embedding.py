from .schema import Chunk


def encode_chunks(chunks: list[Chunk], model):
    return model.encode([chunk.text for chunk in chunks])
