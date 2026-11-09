from pathlib import Path
from dotenv import dotenv_values
from openai import OpenAI
from sentence_transformers import SentenceTransformer, CrossEncoder

from RAG.pipeline.storage import load_index, load_bm25_index, load_source_map
from RAG.pipeline.query_pipeline import retrieve
from RAG.pipeline.prompt_builder import build_prompt
from RAG.pipeline.generator import generate_answer


def answer_one_question(
    query,
    embeddings,
    chunks,
    bm25_index,
    embedding_model: SentenceTransformer,
    reranker_model: CrossEncoder,
    client: OpenAI,
    llm_model,
):

    # TODO 1: 调用 retrieve()
    # retrieval_k=20, final_k=5
    results = retrieve(
        query,
        embeddings,
        chunks,
        bm25_index,
        embedding_model,
        reranker_model,
        retrieval_k=20,
        final_k=5,
    )

    # TODO 2: 根据检索结果生成 Prompt
    prompt = build_prompt(query, results)

    # TODO 3: 调用 DeepSeek
    answer = generate_answer(prompt, client, llm_model)

    # TODO 4: 输出答案
    print(answer)

    # TODO 5: 打印引用编号与 KB ID 的对应关系
    for i, chunk in enumerate(results, start=1):
        print(f"[{i}]\tChunk ID: {chunk.chunk_id}")
