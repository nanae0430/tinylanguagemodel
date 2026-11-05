from openai import OpenAI


def generate_answer(
    prompt: str,
    client: OpenAI,
    model: str,
) -> str:

    if not prompt.strip():
        raise ValueError("Prompt cannot be empty")

    # TODO 1:
    response = client.chat.completions.create(
        model=model or "deepseek-flash",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )
    # TODO 2:
    answer = response.choices[0].message.content

    # TODO 3:
    if not answer.strip():
        raise ValueError("empty response")

    # TODO 4:
    return answer
