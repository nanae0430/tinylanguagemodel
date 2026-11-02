from .embedding import encode_chunks
from .schema import Chunk, Document
from .chunker import chunk_document
from .storage import save_index, load_index
from .loader import load_documents
from sentence_transformers import SentenceTransformer


def build_index(
    source_dir: str,
    index_dir: str,
    model: SentenceTransformer,
    chunk_size: int = 500,
    overlap: int = 100,
) -> None:
    documents = load_documents(source_dir)
    chunks = [].extend(
        chunk_document(document, chunk_size, overlap) for document in documents
    )
    embeddings = encode_chunks(chunks, model)
    save_index(embeddings, chunks, index_dir)
