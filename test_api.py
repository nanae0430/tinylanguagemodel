from pathlib import Path
from dotenv import dotenv_values
from openai import OpenAI

# 读取项目根目录下的 .env
config = dotenv_values(Path(__file__).resolve().parent / ".env")

api_key = config.get("DEEPSEEK_API_KEY")
base_url = config.get("DEEPSEEK_BASE_URL")
model = config.get("DEEPSEEK_MODEL")

if not api_key:
    raise ValueError("Missing DEEPSEEK_API_KEY in .env")

# 创建客户端
client = OpenAI(
    api_key=api_key,
    base_url=base_url or "https://api.deepseek.com",
)

# 调用 DeepSeek
response = client.chat.completions.create(
    model=model or "deepseek-flash",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful AI assistant.",
        },
        {
            "role": "user",
            "content": "请用一句话解释什么是 RAG。",
        },
    ],
    extra_body={"thinking": {"type": "disabled"}},
)

print("回答：", response.choices[0].message.content)

print("Input tokens:", response.usage.prompt_tokens)
print("Output tokens:", response.usage.completion_tokens)

from RAG.pipeline.generator import generate_answer

prompt = """
Context:
[1] RAG combines retrieval with language model generation.

Question:
What is RAG?

Answer using the context and cite your source.
"""

answer = generate_answer(
    prompt=prompt,
    client=client,
    model=model,
)

print("Generated answer:", answer)

assert isinstance(answer, str)
assert len(answer.strip()) > 0

print("Generator test passed!")
