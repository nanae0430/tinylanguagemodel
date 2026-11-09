"""Build an evidence-level index for the 45-question RAG evaluation set.

Run from the repository root:
    python -m pip install jieba
    python build_qa45_index.py

The input JSONL is read-only. This script does NOT re-chunk evidence records.
Each original KBxxx is retained as Chunk.chunk_id, and Sxx as Chunk.doc_id.
"""
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

from RAG.pipeline.schema import Chunk
from RAG.pipeline.embedding import encode_chunks
from RAG.pipeline.storage import save_index, save_bm25_index, save_source_map
from RAG.rag import tokenize, build_bm25_index

SOURCE = Path("RAG/eval_45/knowledge_base.jsonl")
INDEX_DIR = Path("RAG/index_qa45")
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_evidence(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Knowledge base not found: {path}")

    chunks = []
    sources = {}
    metadata = {}
    required = {"chunk_id", "source_id", "source_title", "source_url", "text"}

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            missing = required - record.keys()
            if missing:
                raise ValueError(f"Line {line_number}: missing {sorted(missing)}")

            chunk_id = str(record["chunk_id"]).strip()
            source_id = str(record["source_id"]).strip()
            text = record["text"]
            if not chunk_id or not source_id or not isinstance(text, str) or not text.strip():
                raise ValueError(f"Line {line_number}: empty ID or text")
            if chunk_id in metadata:
                raise ValueError(f"Line {line_number}: duplicate chunk_id {chunk_id}")

            source_url = str(record["source_url"] or "").strip()
            source_title = str(record["source_title"] or "").strip()
            source = source_url or source_title
            if source_id in sources and sources[source_id] != source:
                raise ValueError(f"Line {line_number}: conflicting metadata for {source_id}")
            sources[source_id] = source

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=source_id,
                    start_char=0,
                    end_char=len(text),
                    text=text,
                )
            )
            metadata[chunk_id] = {
                key: value for key, value in record.items() if key != "text"
            }

    if not chunks:
        raise ValueError(f"No evidence records in {path}")
    return chunks, sources, metadata


def main():
    chunks, sources, metadata = load_evidence(SOURCE)

    # Both document indexing and later sparse_search() call the SAME tokenize().
    bm25_index = build_bm25_index([tokenize(chunk.text) for chunk in chunks])

    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = encode_chunks(chunks, model)

    save_index(embeddings, chunks, str(INDEX_DIR))
    save_bm25_index(bm25_index, str(INDEX_DIR))
    save_source_map(sources, str(INDEX_DIR))

    metadata_file = INDEX_DIR / "evidence_metadata.json"
    metadata_file.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Evidence chunks: {len(chunks)}")
    print(f"Original sources: {len(sources)}")
    print(f"Embedding shape: {tuple(embeddings.shape)}")
    print(f"Index directory: {INDEX_DIR}")
    print("Sample chunk IDs:", [chunk.chunk_id for chunk in chunks[:5]])
    print("QA45 index build completed!")


if __name__ == "__main__":
    main()
