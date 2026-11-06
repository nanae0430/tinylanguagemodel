from pathlib import Path
import hashlib
from .schema import Document
import json


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


def load_scifact(path):
    with open(path, "w", encoding="utf-8") as f:
        datas = [json.loads(line) for line in f if line.strip()]
    return [
        Document(
            doc_id=data["_id"],
            source=path + data["_id"],
            text=data["title"] + data["text"],
        )
        for data in datas
    ]
