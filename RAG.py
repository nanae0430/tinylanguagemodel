import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

torch.manual_seed(39)


def cos_similarity(query, docs, top_k):
    cos_query = F.normalize(query, p=2, dim=-1)
    cos_docs = F.normalize(docs, p=2, dim=-1)
    score = cos_query @ cos_docs.T
    return torch.topk(score, top_k, -1)


if __name__ == "__main__":
    # query = torch.randn(4, 768)
    # docs = torch.randn(1000, 768)
    # top_k = 3
    # print(cos_similarity(query, docs, top_k=top_k))

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
    answer = torch.tensor([0, 0, 0, 1, 2, 2], dtype=torch.long, device="cuda")
    doc_embeddings = model.encode(docs, convert_to_tensor=True)
    query_embeddings = model.encode(query, convert_to_tensor=True)

    print(doc_embeddings.shape)
    print(query_embeddings.shape)

    values, indices = cos_similarity(query_embeddings, doc_embeddings, len(docs))
    correct = indices[:, 0] == answer
    print(f"cos_similarity:{values}")
    print(f"index:{indices}")
    print(f"Recall@1:{correct.sum()/len(answer)}")
