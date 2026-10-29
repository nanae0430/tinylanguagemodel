from RAG.pipeline.schema import Document
from RAG.pipeline.chunker import chunk_document

doc = Document("001", "test.txt", "ABCDEFGHIJK")

chunks = chunk_document(doc, chunk_size=5, overlap=2)

for chunk in chunks:
    print(chunk)
