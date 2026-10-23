import torch, math
import torch.nn.functional as F
import re
from collections import Counter, defaultdict

torch.manual_seed(39)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())


def build_inverted_index(docs: list[list[str]]) -> dict[str, dict[int, int]]:
    index = defaultdict(dict)
    for i in range(len(docs)):
        for term, frequency in Counter(docs[i]).items():
            index[term][i] = frequency
    return index


def build_bm25_index(docs: list[list[str]]) -> dict:
    inverted_index = build_inverted_index(docs)

    # TODO: 由你计算以下信息
    num_docs = len(docs)
    doc_lengths = [0] * num_docs
    for term, frequency in inverted_index.items():
        for index, freq in frequency:
            doc_lengths[index] += freq

    avgdl = sum(doc_lengths) / num_docs
    idf_scores = {
        term: math.log(
            (1 + num_docs - len(inverted_index.get(term, {})) + 0.5)
            / (len(inverted_index.get(term, {})) + 0.5)
        )
        for term, frequency in inverted_index.items()
    }

    return {
        "inverted_index": inverted_index,
        "num_docs": num_docs,
        "doc_lengths": doc_lengths,
        "avgdl": avgdl,
        "idf_scores": idf_scores,
    }


def cos_similarity(query: torch.Tensor, docs: torch.Tensor, top_k):
    cos_query = F.normalize(query, p=2, dim=-1)
    cos_docs = F.normalize(docs, p=2, dim=-1)
    score = cos_query @ cos_docs.T
    return torch.topk(score, top_k, -1)


def test_rag():
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    docs = [
        "LoRA通过冻结预训练模型权重，仅训练低秩矩阵来减少微调参数量。",
        "RoPE通过旋转位置编码向注意力机制引入位置信息。",
        "KV Cache缓存历史Key和Value，减少自回归生成时的重复计算。",
    ]

    query = [
        "LoRA 如何减少微调参数？",
        "为什么使用两个较小的矩阵就能够降低模型训练成本？",
        "How does low-rank adaptation reduce trainable parameters?",
        "旋转位置编码有什么作用？",
        "自回归生成时，如何避免重复计算历史 Key 和 Value？",
        "How can we reuse previously computed attention states during decoding?",
    ]
    doc_embeddings = model.encode(docs, convert_to_tensor=True)
    query_embeddings = model.encode(query, convert_to_tensor=True)
    answer = torch.tensor(
        [0, 0, 0, 1, 2, 2], dtype=torch.long, device=query_embeddings.device
    )

    print(doc_embeddings.shape)
    print(query_embeddings.shape)

    values, indices = cos_similarity(query_embeddings, doc_embeddings, len(docs))
    correct = indices[:, 0] == answer
    print(f"cos_similarity:{values}")
    print(f"index:{indices}")
    print(f"Recall@1:{correct.sum()/len(answer)}")


def idf(term, docs: list[list[str]]):
    dft = 0
    for tokens in docs:
        if term in tokens:
            dft += 1
    return math.log((1 + (len(docs) - dft + 0.5) / (dft + 0.5)))


def test_idf():
    docs = [
        "lora uses low rank matrices",
        "rope uses rotary embeddings",
        "kv cache stores key and value",
        "lora reduces trainable parameters",
    ]
    terms = ["lora", "uses"]
    docs = [doc.split() for doc in docs]
    for term in terms:
        print(f"{idf(term, docs):.4f}")


def tf(term, tokens: list[str], avgdl, k1=1.5, b=0.75):
    tf_ = tokens.count(term)
    doc_len = len(tokens)
    return tf_ * (k1 + 1) / (tf_ + k1 * (1 - b + b * doc_len / avgdl))


def test_tf():
    k1 = 1.5
    b = 0.75
    test = [[3, 100, 100], [0, 100, 100], [3, 200, 100]]
    for tf, doc_len, avgdl in test:
        tfscore = tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc_len / avgdl))

        print(tfscore)


def bm25(query: list[str], docs: list[list[str]], k1=1.5, b=0.75):

    bm25_scores = []

    idf_score = {term: idf(term, docs) for term in query}
    avgdl = sum([len(token) for token in docs]) / len(docs)
    for tokens in docs:
        bm25_score = 0
        for term in query:
            tf_score = tf(term, tokens, avgdl, k1, b)
            bm25_score += idf_score[term] * tf_score
        bm25_scores.append(bm25_score)
    return bm25_scores


def test_bm25():
    docs = [
        "lora uses low rank matrices",
        "rope uses rotary embeddings",
        "kv cache stores key and value",
        "lora reduces trainable parameters",
    ]

    query = "lora parameters"
    bm25_scores = bm25(query, docs)
    print(bm25_scores)


def eval_rag(model, retrieval_k, final_k=3):
    device = next(model.parameters()).device
    from retrieval_eval import docs, queries, labels

    bm25_scores = torch.tensor([bm25(query, docs) for query in queries], device=device)
    bm25_values, bm25_indices = torch.topk(bm25_scores, k=retrieval_k, dim=-1)

    queries_tensor = model.encode(queries, convert_to_tensor=True)
    docs_tensor = model.encode(docs, convert_to_tensor=True)
    dr_values, dr_indices = cos_similarity(
        queries_tensor, docs_tensor, top_k=retrieval_k
    )
    rrf_score = rrf(bm25_indices, dr_indices, bm25_values, len(docs))
    rrf_values, rrf_indices = torch.topk(rrf_score, final_k)
    labels = torch.tensor(labels, device=device)
    bm25_correct = (
        (
            (bm25_indices[:, :final_k] == labels[:, None])
            & (bm25_values[:, :final_k] > 0)
        )
        .any(dim=-1)
        .to(device)
    )
    dr_correct = (dr_indices[:, :final_k] == labels[:, None]).any(dim=-1).to(device)
    rrf_correct = (rrf_indices[:, :final_k] == labels[:, None]).any(dim=-1).to(device)
    print(f"bm25 recall@{final_k}:\t{bm25_correct.float().mean().item()}")
    print(f"dr recall@{final_k}:\t{dr_correct.float().mean().item()}")
    print(f"rrf recall@{final_k}:\t{rrf_correct.float().mean().item()}")
    for i in range(len(labels)):
        if not bm25_correct[i]:
            print(
                f"bm25\n"
                f"Query {i}: {queries[i]}\n"
                f"Expected: {labels[i].item()}\n"
                f"Retrieved: {bm25_indices[i].tolist()}"
            )
        if not dr_correct[i]:
            print(
                f"dense\n"
                f"Query {i}: {queries[i]}\n"
                f"Expected: {labels[i].item()}\n"
                f"Retrieved: {dr_indices[i].tolist()}"
            )
        if not rrf_correct[i]:
            print(
                "rrf\n"
                f"query {i}: {queries[i]}\n"
                f"Expected: {labels[i].item()}\n"
                f"Retrieved: {rrf_indices[i].tolist()}"
            )


def rrf(
    bm25_indices: torch.Tensor,
    dense_indices: torch.Tensor,
    bm25_values: torch.Tensor,
    num_docs,
    c=60,
):
    B, T = bm25_indices.shape
    rrf_score = torch.zeros(size=(B, num_docs), device=bm25_indices.device)
    rank = torch.arange(1, 1 + T, device=bm25_indices.device)
    weight = 1 / (c + rank)
    valid_bm25 = bm25_values > 0
    rrf_score.scatter_add_(-1, bm25_indices, valid_bm25.float() * weight[None, :])
    rrf_score.scatter_add_(-1, dense_indices, weight[None, :].expand(B, -1))
    return rrf_score


if __name__ == "__main__":
    # query = torch.randn(4, 768)
    # docs = torch.randn(1000, 768)
    # top_k = 3
    # print(cos_similarity(query, docs, top_k=top_k))
    # test_idf()
    # test_tf()
    # test_bm25()
    # from sentence_transformers import SentenceTransformer

    # model = SentenceTransformer(
    #     "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    # ).to("cuda" if torch.cuda.is_available() else "cpu")
    # for retrieval_k in [1, 3, 5, 8]:
    #     print("*" * 10, f"retrieval_k:{retrieval_k}", "*" * 10)
    #     eval_rag(model, retrieval_k=retrieval_k, final_k=1)
    docs = [["lora", "lora", "training"], ["lora", "model"], ["rope", "model"]]

    print(dict(build_inverted_index(docs)))
