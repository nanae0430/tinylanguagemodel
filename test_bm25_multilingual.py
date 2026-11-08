"""Smoke test for English / Chinese mixed-text BM25 retrieval.

Run from the repository root after: python -m pip install jieba
"""
from RAG.rag import bm25_search, build_bm25_index, tokenize


def main():
    # English-only behavior remains compatible with the existing SciFact benchmark.
    assert tokenize("LS-DYNA and ALE-based simulations.") == [
        "ls-dyna", "and", "ale-based", "simulations"
    ]

    docs = [
        "LS-DYNA 使用 ALE 方法进行网格变形仿真。",
        "有限元模型可以分析材料的应力响应。",
        "Python 可以用于处理实验数据。",
    ]
    query = "如何在 LS-DYNA 中使用 ALE 模拟网格变形？"

    tokenized_docs = [tokenize(doc) for doc in docs]
    query_tokens = tokenize(query)

    print("Query tokens:", query_tokens)
    print("First document tokens:", tokenized_docs[0])

    assert "ls-dyna" in query_tokens
    assert "ale" in query_tokens
    assert any(any("\u3400" <= ch <= "\u9fff" for ch in term) for term in query_tokens)

    index = build_bm25_index(tokenized_docs)
    scores = bm25_search(query_tokens, index)
    best_doc = max(range(len(scores)), key=scores.__getitem__)

    print("BM25 scores:", scores)
    print("Best document:", docs[best_doc])
    assert best_doc == 0, "Expected the ALE / LS-DYNA document to rank first"
    print("Chinese-English BM25 test passed!")


if __name__ == "__main__":
    main()
