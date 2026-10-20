"""Multi-relevance evaluation for SciFact mini/full, with stable document IDs.

Example:
  from evaluation import RetrievalDataset, metrics
  data = RetrievalDataset('scifact_benchmark/mini', split='test')
  # Produce `ranked_doc_ids` yourself, one ordered list of doc IDs per query:
  # result = metrics(ranked_doc_ids, data.gold_sets, ks=(1,3,5,10))
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
from typing import Sequence


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(row) for row in f if row.strip()]


class RetrievalDataset:
    def __init__(self, folder: str, split: str = "test"):
        if split not in ("all", "dev", "test"):
            raise ValueError("split must be 'all', 'dev', or 'test'")
        base = Path(folder)
        self.docs = read_jsonl(base / "corpus.jsonl")
        all_queries = read_jsonl(base / "queries.jsonl")
        self.queries = [q for q in all_queries if split == "all" or q["split"] == split]
        self.doc_ids = [d["_id"] for d in self.docs]
        self.query_ids = [q["_id"] for q in self.queries]
        self.doc_texts = [(d.get("title", "") + "\n" + d.get("text", "")).strip() for d in self.docs]
        self.query_texts = [q["text"] for q in self.queries]
        self.doc_id_to_index = {id_: i for i, id_ in enumerate(self.doc_ids)}
        gold = {qid: set() for qid in self.query_ids}
        with (base / "qrels" / f"{split}.tsv").open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                if row["query-id"] in gold and int(row["score"]) > 0:
                    gold[row["query-id"]].add(row["corpus-id"])
        self.gold_sets = [gold[qid] for qid in self.query_ids]
        if not all(self.gold_sets):
            raise ValueError("Some queries have no relevant documents")
        if any(not s.issubset(self.doc_id_to_index) for s in self.gold_sets):
            raise ValueError("Gold document absent from corpus")

    def index_rankings_to_ids(self, rankings):
        """Convert torch [B,K] / numpy array / Python list row indices into original IDs."""
        if hasattr(rankings, "detach"):
            rankings = rankings.detach().cpu().tolist()
        elif hasattr(rankings, "tolist"):
            rankings = rankings.tolist()
        return [[self.doc_ids[int(i)] for i in row] for row in rankings]


def metrics(ranked_doc_ids: Sequence[Sequence[str]], gold_sets: Sequence[set[str]], ks=(1, 3, 5, 10)) -> dict[str, float]:
    if len(ranked_doc_ids) != len(gold_sets) or not ranked_doc_ids:
        raise ValueError("Ranked predictions and gold labels must have identical nonzero number of queries")
    results = {}
    for k in ks:
        if k <= 0:
            raise ValueError("k must be positive")
        recalls, hits, reciprocal_ranks = [], [], []
        for ranking, gold in zip(ranked_doc_ids, gold_sets):
            if not gold:
                raise ValueError("Every query must have at least one labeled relevant doc")
            topk = list(ranking[:k])
            relevant = len(gold.intersection(topk))
            recalls.append(relevant / len(gold))
            hits.append(float(relevant > 0))
            rank = next((j for j, did in enumerate(topk, 1) if did in gold), None)
            reciprocal_ranks.append(0.0 if rank is None else 1.0 / rank)
        results[f"Recall@{k}"] = sum(recalls) / len(recalls)
        results[f"Hit@{k}"] = sum(hits) / len(hits)
        results[f"MRR@{k}"] = sum(reciprocal_ranks) / len(reciprocal_ranks)
    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="scifact_benchmark/mini")
    p.add_argument("--split", default="test", choices=["all", "dev", "test"])
    args = p.parse_args()
    ds = RetrievalDataset(args.data, split=args.split)
    print(f"Documents: {len(ds.docs)}, Queries: {len(ds.queries)}, Total positive pairs: {sum(map(len,ds.gold_sets))}")
    print("Example Query:", ds.query_texts[0])
    print("Gold document IDs:", sorted(ds.gold_sets[0]))
    print("Evaluation helper ready. Import metrics(...) after your own retrievers produce rankings.")
