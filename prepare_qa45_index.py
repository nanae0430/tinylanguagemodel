"""Build a local QA45 index using the existing RAG pipeline.

Run from the repository root after merging the files from qa45_ready_for_rag.zip:
    python -m pip install jieba
    python prepare_qa45_index.py

Does not re-chunk the 38 evidence snippets; retains their original KB IDs.
Writes into RAG/index/qa45 (already covered by the repository .gitignore).
"""
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

from RAG.pipeline.schema import Chunk
from RAG.pipeline.embedding import encode_chunks
from RAG.pipeline.storage import save_index, save_bm25_index, save_source_map
from RAG.rag import build_bm25_index, tokenize

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "RAG" / "eval_45"
INDEX_DIR = ROOT / "RAG" / "index" / "qa45"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def main():
    data = json.loads((DATA_DIR / "chunks.json").read_text(encoding="utf-8"))
    source_map = json.loads((DATA_DIR / "sources.json").read_text(encoding="utf-8"))
    meta = json.loads((DATA_DIR / "evidence_metadata.json").read_text(encoding="utf-8"))
    chunks = [Chunk(**entry) for entry in data]

    if len({x.chunk_id for x in chunks}) != len(chunks):
        raise ValueError("Duplicate evidence chunk_id")
    if any(x.doc_id not in source_map or x.end_char != len(x.text) for x in chunks):
        raise ValueError("Invalid source mapping or chunk offsets")
    if set(meta) != {x.chunk_id for x in chunks}:
        raise ValueError("Evidence metadata does not match chunk IDs")

    print(f"Indexing {len(chunks)} evidence chunks from {len(source_map)} sources")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = encode_chunks(chunks, model)
    bm25 = build_bm25_index([tokenize(x.text) for x in chunks])

    save_index(embeddings, chunks, str(INDEX_DIR))
    save_bm25_index(bm25, str(INDEX_DIR))
    save_source_map(source_map, str(INDEX_DIR))
    (INDEX_DIR / "evidence_metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Embedding shape:", tuple(embeddings.shape))
    print("Index saved to:", INDEX_DIR)
    print("QA45 index ready!")


if __name__ == "__main__":
    main()
