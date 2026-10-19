import torch, math
import torch.nn.functional as F

torch.manual_seed(39)


def cos_similarity(query, docs, top_k):
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


def bm25(query: str, docs: list[str], k1=1.5, b=0.75):
    terms = query.split()
    bm25_scores = []
    docs_ = [doc.split() for doc in docs]
    idf_score = {term: idf(term, docs_) for term in terms}
    avgdl = sum([len(token) for token in docs_]) / len(docs_)
    for tokens in docs_:
        bm25_score = 0
        for term in terms:
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


def eval_rag(model, k=3):
    device = next(model.parameters()).device
    from retrieval_eval import docs, queries, labels

    bm25_scores = torch.tensor([bm25(query, docs) for query in queries], device=device)
    bm25_values, bm25_indices = torch.topk(bm25_scores, k=k, dim=-1)

    queries_tensor = model.encode(queries, convert_to_tensor=True)
    docs_tensor = model.encode(docs, convert_to_tensor=True)
    dr_values, dr_indices = cos_similarity(queries_tensor, docs_tensor, top_k=k)
    rrf_score = rrf(bm25_indices, dr_indices, bm25_values, len(docs))
    rrf_values, rrf_indices = torch.topk(rrf_score, k)
    labels = torch.tensor(labels, device=device)
    bm25_correct = (bm25_indices == labels[:, None]).any(dim=-1).to(device)
    dr_correct = (dr_indices == labels[:, None]).any(dim=-1).to(device)
    rrf_correct = (rrf_indices == labels[:, None]).any(dim=-1).to(device)
    print(f"bm25 recall@{k}:\t{bm25_correct.float().mean().item()}")
    print(f"dr recall@{k}:\t{dr_correct.float().mean().item()}")
    print(f"rrf recall@{k}:\t{rrf_correct.float().mean().item()}")
    for i in range(len(labels)):
        if not bm25_correct[i]:
            print(
                f"bm25"
                f"Query {i}: {queries[i]}\n"
                f"Expected: {labels[i].item()}\n"
                f"Retrieved: {bm25_indices[i].tolist()}"
            )
        if not dr_correct[i]:
            print(
                f"dense"
                f"Query {i}: {queries[i]}\n"
                f"Expected: {labels[i].item()}\n"
                f"Retrieved: {dr_indices[i].tolist()}"
            )
        if not rrf_correct[i]:
            print(
                "rrf"
                f"query {i}: {queries[i]}\n"
                f"Expected: {labels[i].item()}\n"
                f"Retrieved: {dr_indices[i].tolist()}"
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
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    for i in range(1, 4):
        print("*" * 10, i, "*" * 10)
        eval_rag(model, i)
