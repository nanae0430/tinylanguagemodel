import torch, re
from tinylanguagemodel import TinyLanguageModel
from MinBPE import RegexTokenizer

torch.manual_seed(42)


def prepare_sft_sample(prompt_ids, response_ids, eos_id):
    seq = torch.tensor(data=prompt_ids + response_ids + [eos_id], dtype=torch.long)
    x, label = seq[:-1], seq[1:].clone()
    label[: len(prompt_ids) - 1] = -100
    return x, label


def collate_sft_batch(samples: list[tuple[torch.Tensor, torch.Tensor]], pad_id: int):
    batch_size = len(samples)
    max_len = max(x.shape[0] for x, label in samples)
    x = torch.full(size=(batch_size, max_len), fill_value=pad_id, dtype=torch.long)
    label = torch.full(size=(batch_size, max_len), fill_value=-100, dtype=torch.long)
    for i in range(batch_size):

        x[i, : len(samples[i][0])] = samples[i][0]
        label[i, : len(samples[i][0])] = samples[i][1]
    return x, label


def prepare_sft_text_sample(texts: list[str], tokenizer, eos_id, pad_id):
    questions, answers = [], []

    pattern = r"User:[ \t]*(?P<question>.*?)\r?\nAssistant:[ \t]*(?P<answer>.*)"

    for text in texts:
        text = re.sub(r"\r?\n[ \t]*Assistant:", "\nAssistant:", text.strip(), count=1)
        match = re.fullmatch(pattern, text, flags=re.DOTALL)
        if match is None:
            raise ValueError(f"问答格式不正确：{text!r}")
        prompt, response = re.split(
            r"(?<=\nAssistant:)",
            text,
            maxsplit=1,
        )
        questions.append(prompt)
        answers.append(response)

    questions = [tokenizer.encode(question) for question in questions]
    answers = [tokenizer.encode(answer) for answer in answers]

    samples = [
        prepare_sft_sample(question, answer, eos_id)
        for question, answer in zip(questions, answers)
    ]
    return collate_sft_batch(samples, pad_id)


if __name__ == "__main__":
    eos_id = 2000
    pad_id = 2001
    samples = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    texts = """User: What color is the sky on a sunny day?
            Assistant: The sky is blue.
            <|endoftext|>
            User: What sound does a cat make?
            Assistant: A cat says meow.
            <|endoftext|>
            User: What is one plus two?
            Assistant: One plus two is three.
            <|endoftext|>
            User: Lily feels cold. What can she wear?
            Assistant: Lily can wear a warm coat.
            <|endoftext|>
            User: Tom sees his friend fall down. What should he do?
            Assistant: Tom should help his friend get up and ask if they are hurt.
            <|endoftext|>
            User: Tell me a short story about a dog.
            Assistant: A little dog found a red ball in the park. He brought it to his owner, and they played together.
            <|endoftext|>"""
    texts = [part.strip() for part in texts.split("<|endoftext|>") if part.strip()]

    tokenizer = RegexTokenizer().load("tiny_story.model")
    batch_x, batch_labels = prepare_sft_text_sample(texts, tokenizer, eos_id, pad_id)
    checkpoint = torch.load("last_checkpoint_RMSNorm_SwiGLU.pt")
    model_config = checkpoint["model_config"]
    model_state_dict = checkpoint["model_state_dict"]

    model = TinyLanguageModel(**model_config).to(device)
    model.load_state_dict(model_state_dict)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    batch_x, batch_labels = batch_x.to(device), batch_labels.to(device)
    for i in range(1, 51):
        logits, loss = model(batch_x, batch_labels)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
        optimizer.step()
        # if i % 10 == 0:
        # print(f"{loss.item():.4e}")
        # for name, parameter in model.named_parameters():
        #     assert parameter.grad is not None, f"{name}梯度为 None"
        #     assert torch.isfinite(parameter.grad).all().item(), f"{name}梯度为NaN或inf"
        # print("梯度检查通过")
    test = "User: What color is the sky when the weather is sunny?\nAssistant:"
    model.eval()
    test_idx = tokenizer.encode(test, {"<|endoftext|>"})
    test_tensor = torch.tensor(test_idx, dtype=torch.long, device=device).unsqueeze(0)
    result, logits, _, _ = model.generate(test_tensor, 1000, 2000)
    print(tokenizer.decode(result[0].tolist()))
    torch.save(
        {
            "model_config": model.model_config,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        "last_checkpoint_RMSNorm_SwiGLU_sft.pt",
    )
