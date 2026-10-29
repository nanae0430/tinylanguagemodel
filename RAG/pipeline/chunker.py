from .schema import Document, Chunk


def chunk_document(
    doc: Document, chunk_size: int = 500, overlap: int = 100
) -> list[Chunk]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Invalid chunk_size or overlap")
    step = chunk_size - overlap
    chunked_document = []
    doc_len = len(doc.text)
    for start_char in range(0, doc_len, step):
        end_char = min(start_char + chunk_size, doc_len)
        text = doc.text[start_char:end_char]
        doc_id = doc.doc_id
        chunk_id = doc_id + ":" + str(start_char) + ":" + str(end_char)
        chunked_document.append(Chunk(chunk_id, doc_id, start_char, end_char, text))
        if end_char == doc_len:
            break
    return chunked_document
