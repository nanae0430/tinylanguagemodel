from .embedding import encode_chunks
from .schema import Chunk, Document
from .chunker import chunk_document
from .storage import save_index, save_bm25_index
from .loader import load_documents
from sentence_transformers import SentenceTransformer
from RAG.rag import build_bm25_index, tokenize


def build_index(
    source_dir: str,
    index_dir: str,
    model: SentenceTransformer,
    chunk_size: int = 500,
    overlap: int = 100,
) -> None:
    documents = load_documents(source_dir)
    chunks = []
    for document in documents:
        chunks.extend(chunk_document(document, chunk_size, overlap))
    embeddings = encode_chunks(chunks, model)
    save_index(embeddings, chunks, index_dir)
    bm25_index = build_bm25_index([tokenize(chunk.text) for chunk in chunks])
    save_bm25_index(bm25_index, index_dir)
