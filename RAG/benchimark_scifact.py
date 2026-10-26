from evaluation import RetrievalDataset, metrics
from rag import bm25, tokenize, build_bm25_index, bm25_search, cos_similarity
import torch, pprint, time
from sentence_transformers import SentenceTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"
data = RetrievalDataset("scifact_benchmark/mini", split="dev")
k = 10
docs_origin = data.doc_texts
queries_origin = data.query_texts
docs = [tokenize(doc) for doc in docs_origin]
queries = [tokenize(query) for query in queries_origin]
# print(type(docs))  # 1000
# print(type(queries))  # 40
# print(data.gold_sets[0])  # 第一条 Query 的相关文档 ID 集合
# index=build_bm25_index(docs)
# bm25_scores = torch.tensor([bm25_search(query, index) for query in queries], device=device)
start = time.perf_counter()
bm25_index = build_bm25_index(docs)
build_time = time.perf_counter() - start

start = time.perf_counter()
bm25_scores = torch.tensor(
    [bm25_search(query, bm25_index) for query in queries], device=device
)
search_time = time.perf_counter() - start

print(f"Build index: {build_time:.4f}s")
print(f"Search total: {search_time:.4f}s")
print(f"Search per query: {search_time / len(queries) * 1000:.4f}ms")
bm25_values, bm25_indices = torch.topk(bm25_scores, k)
print((bm25_values == 0).any(-1).sum().item())
print((bm25_values == 0).all(-1).sum().item())
bm25_ids = data.index_rankings_to_ids(bm25_indices)
result = metrics(bm25_ids, data.gold_sets)
print(result)
is_hit = []
for retrieved, gold in zip(bm25_ids, data.gold_sets):
    if any(idx in gold for idx in retrieved):
        is_hit.append(True)
    else:
        is_hit.append(False)
missed_index = []
for index, value in enumerate(is_hit):
    if not value:
        missed_index.append(index)
print(missed_index)
for index in missed_index:
    # print(data.docs[bm25_indices[index, 0].item()])
    print("*" * 10, index, "*" * 10)
    print("query:", data.query_texts[index])
    print(f"bm25 top1:\t{bm25_values[index, 0].item()}")
    query_terms = set(queries[index])

    for gold_id in sorted(data.gold_sets[index]):
        j = data.doc_id_to_index[gold_id]
        doc_terms = set(docs[j])

        print("Gold ID:", gold_id)
        print("Gold Score:", bm25_scores[index, j].item())
        print("Matched Terms:", query_terms & doc_terms)
        print("Gold Document:", data.doc_texts[j][:600])
    gold_id = next(iter(data.gold_sets[index]))
    gold_index = data.doc_id_to_index[gold_id]
    print(f"gold index:\t{bm25_scores[index, gold_index].item()}")

    # print(data.docs[gold_index])
print("*" * 10 + "dense retrieval model" + "*" * 10)
model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
).to(device)
queries_tensor = model.encode(queries_origin, convert_to_tensor=True)
docs_tensor = model.encode(docs_origin, convert_to_tensor=True)
dr_values, dr_indices = cos_similarity(queries_tensor, docs_tensor, top_k=k)
dr_ids = data.index_rankings_to_ids(dr_indices)
result = metrics(dr_ids, data.gold_sets)
print(result)

bm25_only = []
dense_only = []
both_hit = []
both_miss = []

for i, (bm25_docs, dense_docs, gold) in enumerate(
    zip(bm25_ids, dr_ids, data.gold_sets)
):
    bm25_hit = bool(set(bm25_docs) & gold)
    dense_hit = bool(set(dense_docs) & gold)

    if bm25_hit and dense_hit:
        both_hit.append(i)
    elif bm25_hit and not dense_hit:
        bm25_only.append(i)
    elif not bm25_hit and dense_hit:
        dense_only.append(i)
    else:
        both_miss.append(i)

print("BM25 only:", bm25_only)
print("Dense only:", dense_only)
print("Both hit:", both_hit)
print("Both miss:", both_miss)

union_hit = len(bm25_only) + len(dense_only) + len(both_hit)
print(f"Union Hit@10: {union_hit / len(data.gold_sets):.4f}")
