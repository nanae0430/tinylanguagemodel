from dataclasses import dataclass
from pathlib import Path
import hashlib


@dataclass
class Document:
    doc_id: str
    source: str
    text: str


@dataclass
class Chunk:
    chunk_id: str
    start_char: int
    end_char: int


def load_documents(directory: str) -> list[Document]:

    documents = []
    root = Path(directory)
    assert root.exists()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in [".txt", ".md"]:
            continue
        text = path.read_text("utf-8")
        relative_path = path.relative_to(root).as_posix()
        doc_id = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()
        documents.append(Document(doc_id, relative_path, text))
    return documents


documents = load_documents(r"RAG\test_data")
print(documents)
