"""Offline integration test with tiny fabricated in-memory ZIP fixture; source data not included."""

from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile
import json
import scifact_benchmark.build_dataset as build_dataset
from scifact_benchmark.evaluation import RetrievalDataset, metrics


def check():
    with TemporaryDirectory() as d:
        source = Path(d) / "fixture.zip"
        rows = [
            {"_id": str(i), "title": f"Title {i}", "text": f"Scientific abstract {i}"}
            for i in range(20)
        ]
        q = [{"_id": str(i), "text": f"Claim {i}"} for i in range(10)]
        qrel = "query-id\tcorpus-id\tscore\n" + "".join(
            f"{i}\t{i}\t1\n" for i in range(10)
        )
        with ZipFile(source, "w") as z:
            z.writestr("scifact/corpus.jsonl", "\n".join(json.dumps(x) for x in rows))
            z.writestr("scifact/queries.jsonl", "\n".join(json.dumps(x) for x in q))
            z.writestr("scifact/qrels/test.tsv", qrel)
        dest = Path(d) / "out"
        build_dataset.build(
            source,
            dest,
            num_docs=12,
            num_queries=6,
            dev_queries=2,
            seed=39,
            include_full=True,
        )
        mini = RetrievalDataset(str(dest / "mini"), split="all")
        assert len(mini.docs) == 12 and len(mini.queries) == 6
        assert len(mini.gold_sets) == 6 and all(len(x) == 1 for x in mini.gold_sets)
        full = RetrievalDataset(str(dest / "full"), split="all")
        assert len(full.docs) == 20 and len(full.queries) == 10
        rankings = [[next(iter(g))] for g in mini.gold_sets]
        met = metrics(rankings, mini.gold_sets, ks=(1, 3))
        assert met["Recall@1"] == met["Hit@1"] == met["MRR@1"] == 1
        print("PASS: source parsing, mini/full generation, multi-relevance metric API")


if __name__ == "__main__":
    check()
