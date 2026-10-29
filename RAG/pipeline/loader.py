from pathlib import Path
import hashlib
from schema import Document


def load_documents(directory: str) -> list[Document]:

    documents = []
    root = Path(directory)
    if not root.is_dir():
        raise NotADirectoryError
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in [".txt", ".md"]:
            continue
        text = path.read_text("utf-8")
        relative_path = path.relative_to(root).as_posix()
        doc_id = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()
        documents.append(Document(doc_id, relative_path, text))
    return documents
