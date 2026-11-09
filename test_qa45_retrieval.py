"""Smoke-test and evaluate KB-id retrieval on 45-question evaluation corpus.

    python test_qa45_retrieval.py --qid Q009
    python test_qa45_retrieval.py --all

Only Q001-Q037 have positive qrels; Q038-Q045 are abstention cases.
Retrieval metrics are computed over the 37 answerable cases, NOT all 45.
"""

import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict

from sentence_transformers import SentenceTransformer, CrossEncoder
from RAG.pipeline.storage import load_index, load_bm25_index
from RAG.pipeline.query_pipeline import retrieve

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "RAG" / "eval_45"
INDEX = ROOT / "RAG" / "index" / "qa45"


def read_questions():
    with (DATA / "qa_eval_45_questions.jsonl").open(encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_qrels():
    gold = defaultdict(set)
    with (DATA / "qrels.tsv").open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if int(row["relevance"]) > 0:
                gold[row["query_id"]].add(row["chunk_id"])
    return gold


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--qid", default="Q009", help="Single question ID, e.g. Q009")
    group.add_argument(
        "--all",
        action="store_true",
        help="Run all 45 questions; score positive qrels only",
    )
    args = parser.parse_args()

    questions = read_questions()
    questions = questions if args.all else [q for q in questions if q["id"] == args.qid]
    if not questions:
        raise ValueError(f"Unknown question ID: {args.qid}")

    embeddings, chunks = load_index(str(INDEX))
    bm25 = load_bm25_index(str(INDEX))
    embed_model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    qrels = read_qrels()

    scores = {1: [], 3: [], 5: [], 10: []}
    for q in questions:
        # Small corpus: use 20 initial candidates, rerank 5 (same retrieve() function).
        results = retrieve(
            q["question"],
            embeddings,
            chunks,
            bm25,
            embed_model,
            reranker,
            retrieval_k=20,
            final_k=5,
        )
        ids = [chunk.chunk_id for chunk, _ in results]
        gold = qrels.get(q["id"], set())
        print(f"{q['id']} | {q['question']}")
        print("Top5 KB IDs:", ids)
        print(
            "Gold IDs:   ",
            sorted(gold) if gold else "No positive qrels (abstention case)",
        )
        if gold:
            for k in scores:
                scores[k].append(len(set(ids[:k]) & gold) / len(gold))

    if args.all:
        print("\nRecall@K (37 answerable queries only):")
        for k, values in scores.items():
            print(f"Recall@{k}: {sum(values)/len(values):.4f} (n={len(values)})")
        print(
            "Note: This does not evaluate generation, citation faithfulness, or refusal."
        )


if __name__ == "__main__":
    main()
