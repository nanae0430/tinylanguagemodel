from pathlib import Path
from dotenv import dotenv_values
from openai import OpenAI
from sentence_transformers import SentenceTransformer, CrossEncoder

from RAG.pipeline.storage import load_index, load_bm25_index, load_source_map
from RAG.pipeline.query_pipeline import retrieve
from RAG.pipeline.prompt_builder import build_prompt, DEFAULT_INSTRUCTIONS
from RAG.pipeline.generator import generate_answer

import json, re, csv


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
        query["question"],
        embeddings,
        chunks,
        bm25_index,
        embedding_model,
        reranker_model,
        retrieval_k=20,
        final_k=5,
    )

    # TODO 2: 根据检索结果生成 Prompt
    prompt = build_prompt(
        query["question"],
        results,
        instruction=DEFAULT_INSTRUCTIONS
        + """\n6. If the provided evidence is insufficient to answer the question,
explicitly abstain. Do not cite irrelevant evidence.""",
    )

    # TODO 3: 调用 DeepSeek
    answer = generate_answer(prompt, client, llm_model)

    # TODO 4: 输出答案
    print(f"{query['id']}\t{query['question']}\n\nanswer:{answer}\n")

    # TODO 5: 打印引用编号与 KB ID 的对应关系
    for i, (chunk, score) in enumerate(results, start=1):
        print(f"[{i}]\tChunk ID: {chunk.chunk_id}\tScore: {score:.4f}")
    return answer, results


def main():
    # 1. 加载 QA45 索引（RAG/index/qa45）
    embeddings, chunks = load_index("RAG/index/qa45")
    bm25_index = load_bm25_index("RAG/index/qa45")
    source_map = load_source_map("RAG/index/qa45")
    # 2. 初始化 SentenceTransformer 和 CrossEncoder
    embedding_model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    # 3. 读取 .env，创建 DeepSeek Client
    config = dotenv_values(Path(__file__).resolve().parent / ".env")

    api_key = config.get("DEEPSEEK_API_KEY")
    base_url = config.get("DEEPSEEK_BASE_URL")
    model = config.get("DEEPSEEK_MODEL")

    if not api_key:
        raise ValueError("Missing DEEPSEEK_API_KEY in .env")

    client = OpenAI(api_key=api_key, base_url=base_url or "https://api.deepseek.com")

    # 4. 调用 answer_one_question()
    with open(
        r"RAG\LLM_RAG_QA_Eval_45\qa_eval_45_questions.jsonl", "r", encoding="utf-8"
    ) as f:
        queries = [json.loads(line) for line in f if line.strip()]
    qa = []
    for query in queries:
        answer, results = answer_one_question(
            query,
            embeddings,
            chunks,
            bm25_index,
            embedding_model,
            reranker_model,
            client,
            llm_model=model or "deepseek-flash",
        )
        answer_text = re.sub(r"\[\d+\]", "", answer).strip()

        abstained = answer_text.startswith(
            (
                "我不知道",
                "无法确定",
                "无法从",
                "不能确定",
                "资料不足",
                "提供的上下文不足",
            )
        )
        citation_nums = re.findall(r"\[(\d+)\]", answer)
        citations = [results[int(i) - 1][0].chunk_id for i in citation_nums]
        qa.append(
            {
                "id": query["id"],
                "question": query["question"],
                "answer": answer,
                "abstained": abstained,
                "citations": citations,
                "retrieval_ids": [chunk.chunk_id for chunk, _ in results],
            }
        )

    answerable_ids = set()

    with open("RAG/LLM_RAG_QA_Eval_45/qrels.tsv", "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            if int(row["relevance"]) > 0:
                answerable_ids.add(row["query_id"])
    gold_abstained = query["id"] not in answerable_ids
    correct_refusal = abstained and gold_abstained
    false_refusal = abstained and not gold_abstained
    missed_refusal = not abstained and gold_abstained
    print(f"correct_refusal:\n{correct_refusal}")
    print(f"false_refusal:\n{false_refusal}")
    print(f"missed_refusal:\n{missed_refusal}")
    output_dir = Path("RAG/eval_45")
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for item in qa:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def test_refusal():
    answerable_ids = set()

    with open("RAG/LLM_RAG_QA_Eval_45/qrels.tsv", "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            if int(row["relevance"]) > 0:
                answerable_ids.add(row["query_id"])
    with open("RAG/eval_45/predictions.jsonl", "r", encoding="utf-8") as f:
        qa = [json.loads(line) for line in f if line.strip()]
    stats = {
        "correct_refusal": 0,
        "false_refusal": 0,
        "missed_refusal": 0,
        "answered": 0,
    }

    for item in qa:
        gold_abstained = item["id"] not in answerable_ids
        abstained = item["abstained"]

        if abstained and gold_abstained:
            stats["correct_refusal"] += 1
        elif abstained and not gold_abstained:
            stats["false_refusal"] += 1
        elif not abstained and gold_abstained:
            stats["missed_refusal"] += 1
        else:
            stats["answered"] += 1

    print(stats)


if __name__ == "__main__":
    test_refusal()
