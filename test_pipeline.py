from RAG.pipeline.schema import Document
from RAG.pipeline.chunker import chunk_document
from RAG.pipeline.embedding import encode_chunks
from sentence_transformers import SentenceTransformer

doc = Document("001", "test.txt", "ABCDEFGHIJK")

chunks = chunk_document(doc, chunk_size=5, overlap=2)

for chunk in chunks:
    print(chunk)

model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

embeddings = encode_chunks(chunks, model)

print(embeddings.shape)
print(embeddings.dtype)

empty = encode_chunks([], model)
print(empty.shape)

assert embeddings.shape == (len(chunks), 384)
assert empty.shape == (0, 384)
