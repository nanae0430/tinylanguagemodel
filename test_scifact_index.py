import time

from sentence_transformers import SentenceTransformer
from RAG.pipeline.index_builder import build_index
from RAG.pipeline.storage import load_index, load_source_map

model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

start = time.perf_counter()

build_index(
    source_dir="scifact_benchmark/mini/corpus.jsonl",
    index_dir="RAG/index_scifact",
    model=model,
    chunk_size=500,
    overlap=100,
)

elapsed = time.perf_counter() - start

embeddings, chunks = load_index("RAG/index_scifact")
source_map = load_source_map("RAG/index_scifact")

print(f"Build time: {elapsed:.2f}s")
print(f"Documents: {len(source_map)}")
print(f"Chunks: {len(chunks)}")
print(f"Embeddings shape: {embeddings.shape}")

assert len(source_map) == 1000
assert len(chunks) == embeddings.shape[0]
assert all(chunk.doc_id in source_map for chunk in chunks)

print("SciFact Index Builder test passed!")
