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
    doc_id: str
    start_char: int
    end_char: int
    text: str
