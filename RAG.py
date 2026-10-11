import torch
import torch.nn.functional as F

torch.seed(39)


def cos_similarity(query, docs, top_k):
    cos_query = F.normalize(query, 2, -1)
    cos_docs = F.normalize(docs, 2, -1)
    score = cos_query @ cos_docs.T
    return torch.topk(score, top_k, -1)
