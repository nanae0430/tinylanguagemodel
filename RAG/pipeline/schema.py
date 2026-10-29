from dataclasses import dataclass
from pathlib import Path


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

    doc_id = 0
    documents = []
    root = Path(directory)
    assert root.exists()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in [".txt", ".md"]:
            continue
        text = path.read_text("utf-8")
        documents.append(Document(str(doc_id), path, text))
        doc_id += 1
    return documents


documents = load_documents(r"RAG\test_data")
print(documents)
