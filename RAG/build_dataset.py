#!/usr/bin/env python3
"""Build a reproducible BEIR SciFact benchmark: mini (1000 docs / 200 queries) + full.

No third-party Python dependency. Source: official BEIR scifact.zip.

Example:
    python build_dataset.py --output scifact_benchmark
    python build_dataset.py --source C:/Downloads/scifact.zip --output scifact_benchmark
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import shutil
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

SOURCE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"


def load_jsonl(raw: bytes) -> list[dict]:
    return [json.loads(line) for line in raw.decode("utf-8-sig").splitlines() if line.strip()]


def get_zip_member(archive: zipfile.ZipFile, target: str) -> bytes:
    candidates = [x for x in archive.namelist() if x.endswith(target) and not x.endswith("/")]
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one {target} member, found: {candidates}")
    return archive.read(candidates[0])


def get_source(path: Path) -> None:
    if path.exists():
        print(f"Using cached official archive: {path}")
        return
    print(f"Downloading official BEIR SciFact: {SOURCE_URL}")
    req = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "Mozilla/5.0 (SciFact-subset-builder)"})
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".part")
    try:
        with urllib.request.urlopen(req, timeout=90) as response, partial.open("wb") as target:
            shutil.copyfileobj(response, target)
        if not zipfile.is_zipfile(partial):
            raise RuntimeError("Downloaded source is not a valid ZIP")
        partial.replace(path)
    except Exception:
        partial.unlink(missing_ok=True)
        print("Download failed. Manually download scifact.zip from the source URL and use --source PATH.")
        raise


def read_original(zip_path: Path) -> tuple[dict, dict, dict]:
    with zipfile.ZipFile(zip_path) as z:
        corpus_rows = load_jsonl(get_zip_member(z, "corpus.jsonl"))
        query_rows = load_jsonl(get_zip_member(z, "queries.jsonl"))
        raw_qrels = get_zip_member(z, "qrels/test.tsv")
    corpus = {str(doc["_id"]): doc for doc in corpus_rows}
    queries = {str(query["_id"]): query for query in query_rows}
    if len(corpus) != len(corpus_rows) or len(queries) != len(query_rows):
        raise ValueError("Non-unique document/query ID in source")
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    reader = csv.DictReader(io.StringIO(raw_qrels.decode("utf-8-sig")), delimiter="\t")
    for row in reader:
        qid, did = str(row["query-id"]), str(row["corpus-id"])
        score = int(row["score"])
        if score > 0 and qid in queries and did in corpus:
            qrels[qid][did] = score
    if not qrels:
        raise ValueError("No usable positive relevance judgments")
    return corpus, queries, dict(qrels)


def sort_id(x: str):
    try:
        return (0, int(x))
    except ValueError:
        return (1, x)


def dump_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def dump_qrels(path: Path, qrels: dict[str, dict[str, int]], selected_qids: list[str], available: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["query-id", "corpus-id", "score"])
        for qid in selected_qids:
            for did, rel in sorted(qrels[qid].items(), key=lambda item: sort_id(item[0])):
                if did in available:
                    writer.writerow([qid, did, rel])


def make_output(path: Path, corpus: dict, queries: dict, qrels: dict,
                selected_doc_ids: list[str], selected_query_ids: list[str],
                dev_ids: set[str], seed: int, zip_sha256: str, name: str) -> dict:
    path.mkdir(parents=True, exist_ok=True)
    doc_set = set(selected_doc_ids)
    query_rows = [{"_id": qid, "text": queries[qid]["text"],
                   "split": ("dev" if qid in dev_ids else "test")}
                  for qid in selected_query_ids]
    doc_rows = [{"_id": did, "title": corpus[did].get("title", ""),
                 "text": corpus[did].get("text", "")} for did in selected_doc_ids]
    assert len(doc_rows) == len(doc_set) and len(query_rows) == len(set(selected_query_ids))
    assert all(qid in qrels and (set(qrels[qid]) & doc_set) for qid in selected_query_ids)
    dump_jsonl(path / "corpus.jsonl", doc_rows)
    dump_jsonl(path / "queries.jsonl", query_rows)
    dump_qrels(path / "qrels" / "all.tsv", qrels, selected_query_ids, doc_set)
    dump_qrels(path / "qrels" / "dev.tsv", qrels, [x for x in selected_query_ids if x in dev_ids], doc_set)
    dump_qrels(path / "qrels" / "test.tsv", qrels, [x for x in selected_query_ids if x not in dev_ids], doc_set)
    qrels_count = sum(len(set(qrels[qid]) & doc_set) for qid in selected_query_ids)
    stats = {"name": name, "source": "BEIR SciFact, official test qrels",
             "source_url": SOURCE_URL, "source_zip_sha256": zip_sha256,
             "random_seed": seed, "num_docs": len(doc_rows),
             "num_queries": len(query_rows), "num_dev": len(dev_ids),
             "num_test": len(selected_query_ids) - len(dev_ids),
             "num_qrels": qrels_count, "all_qrels_retained": True,
             "notice": "Reduced-corpus results are NOT comparable to official full-corpus BEIR scores."
             if name == "mini" else "Full SciFact corpus; 300 official test queries, split locally for experimentation."}
    (path / "metadata.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return stats


def build(zip_path: Path, out_root: Path, num_docs: int, num_queries: int,
          dev_queries: int, seed: int, include_full: bool) -> None:
    if num_docs < 1 or num_queries < 1 or dev_queries < 0 or dev_queries >= num_queries:
        raise ValueError("Require positive counts and 0 <= dev_queries < num_queries")
    zip_sha = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    corpus, queries, qrels = read_original(zip_path)
    eligible_qids = sorted(qrels.keys(), key=sort_id)
    if num_queries > len(eligible_qids):
        raise ValueError(f"Only {len(eligible_qids)} test queries with qrels; requested {num_queries}")
    rng = random.Random(seed)
    selected_query_ids = rng.sample(eligible_qids, num_queries)
    rng.shuffle(selected_query_ids)
    dev_ids = set(selected_query_ids[:dev_queries])
    required_docs = {did for qid in selected_query_ids for did in qrels[qid]}
    if len(required_docs) > num_docs:
        raise ValueError(f"{len(required_docs)} relevant docs required; increase --num-docs")
    remaining = sorted(set(corpus) - required_docs, key=sort_id)
    fillers = rng.sample(remaining, num_docs - len(required_docs))
    selected_doc_ids = sorted(required_docs, key=sort_id) + fillers
    rng.shuffle(selected_doc_ids)
    mini = make_output(out_root / "mini", corpus, queries, qrels,
                       selected_doc_ids, selected_query_ids, dev_ids, seed, zip_sha, "mini")
    print("MINI:", json.dumps(mini, indent=2))
    if include_full:
        full_qids = sorted(eligible_qids, key=sort_id)
        full_dev_ids = set(rng.sample(full_qids, min(60, len(full_qids) - 1)))
        full_doc_ids = sorted(corpus, key=sort_id)
        full = make_output(out_root / "full", corpus, queries, qrels,
                           full_doc_ids, full_qids, full_dev_ids, seed, zip_sha, "full")
        print("FULL:", json.dumps(full, indent=2))
    print(f"\nCreated benchmark in {out_root.resolve()}")
    print("Important: corpus.jsonl contains title + abstract, not artificial generated snippets.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=Path("scifact.zip"), help="Local original BEIR SciFact ZIP (download automatically if missing)")
    parser.add_argument("--output", type=Path, default=Path("scifact_benchmark"))
    parser.add_argument("--num-docs", type=int, default=1000)
    parser.add_argument("--num-queries", type=int, default=200)
    parser.add_argument("--dev-queries", type=int, default=40)
    parser.add_argument("--seed", type=int, default=39)
    parser.add_argument("--no-full", action="store_true", help="Skip the original full 5183-document benchmark")
    args = parser.parse_args()
    get_source(args.source)
    build(args.source, args.output, args.num_docs, args.num_queries,
          args.dev_queries, args.seed, not args.no_full)


if __name__ == "__main__":
    main()
