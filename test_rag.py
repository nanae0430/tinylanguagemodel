# test_rag.py

from pathlib import Path
from dotenv import dotenv_values
from openai import OpenAI
from sentence_transformers import SentenceTransformer, CrossEncoder

from RAG.pipeline.storage import load_index, load_bm25_index, load_source_map
from RAG.pipeline.query_pipeline import retrieve
from RAG.pipeline.prompt_builder import build_prompt
from RAG.pipeline.generator import generate_answer

# 1. 读取 .env，初始化 DeepSeek Client
config = dotenv_values(Path(__file__).resolve().parent / ".env")

api_key = config.get("DEEPSEEK_API_KEY")
base_url = config.get("DEEPSEEK_BASE_URL")
model = config.get("DEEPSEEK_MODEL")

source_map = load_source_map("RAG/index_smoke")
if not api_key:
    raise ValueError("Missing DEEPSEEK_API_KEY in .env")

client = OpenAI(api_key=api_key, base_url=base_url or "https://api.deepseek.com")

# 2. 初始化 Embedding Model 和 CrossEncoder
embedding_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
# 3. 从 RAG/index_smoke 加载索引
# TODO
embeddings, chunks = load_index("RAG/index_smoke")
bm25_index = load_bm25_index("RAG/index_smoke")

# 4. 用户输入 Query
query = "Who is Miku?"

# 5. 调用 retrieve() 获取 Top-K Chunks
# TODO
topk_chunks = retrieve(
    query, embeddings, chunks, bm25_index, embedding_model, reranker_model
)

# 6. 调用 build_prompt() 构造 Prompt
# TODO
prompt = build_prompt(query, topk_chunks)
# 7. 调用 generate_answer() 获取 LLM 回答
# TODO
answer = generate_answer(prompt, client, model=model or "deepseek-flash")
# 8. 打印答案和检索到的 Chunk
# TODO
print(f"answer:\n{answer}\n\n")
for i, (chunk, score) in enumerate(topk_chunks, start=1):
    print(f"[{i}] {source_map[chunk.doc_id]}")
    print(f"\tChunk ID: {chunk.chunk_id}")
    print(f"\tScore: {score:.4f}")

from RAG.pipeline.citation import validate_citations

is_valid = validate_citations(answer, topk_chunks)

print("Citation valid:", is_valid)
