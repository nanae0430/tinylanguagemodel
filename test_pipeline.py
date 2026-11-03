from RAG.pipeline.schema import Document
from RAG.pipeline.chunker import chunk_document
from RAG.pipeline.embedding import encode_chunks
from sentence_transformers import SentenceTransformer
from RAG.pipeline.storage import save_index, load_index
import torch

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


save_index(embeddings, chunks, "RAG/index")

loaded_embeddings, loaded_chunks = load_index("RAG/index")

assert torch.equal(embeddings.cpu(), loaded_embeddings)
assert chunks == loaded_chunks

print("Storage test passed!")

from RAG.pipeline.index_builder import build_index

build_index(
    source_dir="RAG/test_data",
    index_dir="RAG/index_smoke",
    model=model,
    chunk_size=500,
    overlap=100,
)

embeddings, chunks = load_index("RAG/index_smoke")

assert embeddings.shape[0] == len(chunks)
assert len(chunks) == 3

print("Index Builder test passed!")

from RAG.pipeline.retriever import dense_search

results = dense_search(
    query="Who is Miku?",
    embeddings=embeddings,
    chunks=chunks,
    model=model,
    top_k=5,
)

for chunk, score in results:
    print(chunk.text, score)

assert len(results) == 3
assert all(isinstance(score, float) for _, score in results)

print("Dense Retriever test passed!")

from RAG.rag import tokenize, build_bm25_index
from RAG.pipeline.retriever import sparse_search

bm25_index = build_bm25_index([tokenize(chunk.text) for chunk in chunks])

from RAG.pipeline.storage import save_bm25_index, load_bm25_index

save_bm25_index(bm25_index, "RAG/index_smoke")
loaded_bm25 = load_bm25_index("RAG/index_smoke")

assert bm25_index == loaded_bm25

results = sparse_search(
    query="miku",
    chunks=chunks,
    bm25_index=loaded_bm25,
    top_k=5,
)

assert len(results) == 1
assert results[0][0].text == "i'm miku!!!"

print("BM25 Persistence test passed!")
