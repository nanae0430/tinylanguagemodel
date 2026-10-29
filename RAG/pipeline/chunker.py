from .schema import Document, Chunk


def chunk_document(
    doc: Document, chunk_size: int = 500, overlap: int = 100
) -> list[Chunk]:
    step = chunk_size - overlap
    chunked_document = []
    for start_char in range(0, len(doc.text), step):
        end_char = start_char + chunk_size
        text = doc.text[start_char:end_char]
        doc_id = doc.doc_id
        chunk_id = doc_id + str(start_char) + str(end_char)
        chunked_document.append(Chunk(chunk_id, doc_id, start_char, end_char, text))
    return chunked_document


if __name__ == "main":
    doc = Document(
        doc_id="001",
        source="test.txt",
        text="ABCDEFGHIJK",
    )
    print(chunk_document(doc, 5, 2))
