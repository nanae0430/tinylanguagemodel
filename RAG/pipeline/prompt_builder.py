from .schema import Chunk


def build_prompt(query: str, results: list[tuple[Chunk, float]]):
    if not query.split():
        raise ValueError("query cannot be empty")
    context_parts = []
    for i, (chunk, value) in enumerate(results, start=1):
        context_parts.append(
            f"[{i}]\n\ndoc_id:\n{chunk.doc_id}\nchunk_id:\n{chunk.chunk_id}\ntext:\n{chunk.text}\n\n"
        )
    return (
        "context\n\n" + "".join(context_parts) + f"question:\n{query}\n\n" + "answer:"
    )
