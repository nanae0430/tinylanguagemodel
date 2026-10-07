import torch, re, random
from tinylanguagemodel import TinyLanguageModel, save_model, LoRALinear
from MinBPE import RegexTokenizer

torch.manual_seed(39)

random.seed(39)


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


def get_sft_batch(texts, tokenizer, batch_size, eos_id, pad_id):

    batch_texts = random.sample(texts, k=batch_size)
    return prepare_sft_text_sample(batch_texts, tokenizer, eos_id, pad_id)


def eval_for_sft(
    model,
    train_texts,
    test_texts,
    tokenizer,
    batch_size,
    device,
    eos_id,
    pad_id,
    eval_iters=1,
):
    model.eval()
    train_losses, eval_losses = [], []

    with torch.no_grad():
        for _ in range(eval_iters):
            test_question, test_answer = get_sft_batch(
                test_texts, tokenizer, batch_size, eos_id, pad_id
            )
            train_question, train_answer = get_sft_batch(
                train_texts, tokenizer, batch_size, eos_id, pad_id
            )
            test_question, test_answer = test_question.to(device), test_answer.to(
                device
            )
            train_question, train_answer = train_question.to(device), train_answer.to(
                device
            )
            _, eval_loss = model(test_question, test_answer)
            _, train_loss = model(train_question, train_answer)
            train_losses.append(train_loss.item())
            eval_losses.append(eval_loss.item())
    model.train()
    return sum(train_losses) / len(train_losses), sum(eval_losses) / len(eval_losses)


def train_sft(
    model,
    optimizer,
    tokenizer,
    train_texts,
    test_texts,
    batch_size,
    eos_id,
    pad_id,
    eval_iters=10,
    best_val_loss=float("inf"),
    steps=200,
    start_step=0,
    print_interval=50,
    save_interval=100,
    non_improve=0,
    is_eval=True,
    patience=3,
    min_delta=0.0,
    path="",
):
    device = next(model.parameters()).device
    for step in range(start_step + 1, steps + 1):
        batch_x, batch_labels = get_sft_batch(
            train_texts, tokenizer, batch_size, eos_id, pad_id
        )
        batch_x, batch_labels = batch_x.to(device), batch_labels.to(device)
        logits, loss = model(batch_x, batch_labels)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        if step % print_interval == 0:
            print(f"{step}\ttrain loss:{loss.item():.4e}")
            # for name, parameter in model.named_parameters():
            #     assert parameter.grad is not None, f"{name}梯度为 None"
            #     assert torch.isfinite(parameter.grad).all().item(), f"{name}梯度为NaN或inf"
            # print("梯度检查通过")
        if (step % save_interval == 0 or step == steps) and is_eval:
            train_loss, eval_loss = eval_for_sft(
                model,
                train_texts,
                test_texts,
                tokenizer,
                batch_size,
                device,
                eos_id,
                pad_id,
                eval_iters,
            )
            print(f"{step}\ttrain loss:{train_loss:.4e}\teval loss:{eval_loss:.4e}")
            best_val_loss, non_improve = save_model(
                best_val_loss=best_val_loss,
                current_train_loss=train_loss,
                current_eval_loss=eval_loss,
                min_delta=min_delta,
                model=model,
                non_improve=non_improve,
                optimizer=optimizer,
                step=step,
                path=path,
            )
            if non_improve >= patience:
                return


def apply_lora(model: TinyLanguageModel, target_modules: list[str], r, alpha):
    paths = [target_module.split(".") for target_module in target_modules]
    for block in model.model[:-1]:
        for path in paths:
            parent = block
            for p in path[:-1]:
                parent = getattr(parent, p)
            base_layer = getattr(parent, path[-1])
            if not isinstance(base_layer, torch.nn.Linear):
                raise TypeError(f"目标不是线性层：{'.'.join(path)}")
            setattr(parent, path[-1], LoRALinear(base_layer, r, alpha))


def test_lora_model(
    model: TinyLanguageModel,
    target_modules,
    tokenizer,
    train_texts,
    test_texts,
    r,
    alpha,
    eos_id,
    pad_id,
    lr=1e-4,
    batch_size=2,
    eval_iters=10,
    best_val_loss=float("inf"),
    steps=200,
    start_step=0,
    print_interval=50,
    save_interval=100,
    non_improve=0,
    is_eval=True,
    patience=3,
    min_delta=0.0,
    path="",
):
    model.requires_grad_(False)

    apply_lora(model, target_modules, r, alpha)
    optim_param = [
        parameter for parameter in model.parameters() if parameter.requires_grad is True
    ]
    optimizer = torch.optim.AdamW(optim_param, lr=lr)
    train_sft(
        model,
        optimizer,
        tokenizer,
        train_texts,
        test_texts,
        batch_size,
        eos_id,
        pad_id,
        eval_iters,
        best_val_loss,
        steps,
        start_step,
        print_interval,
        save_interval,
        non_improve,
        is_eval,
        patience,
        min_delta,
        path,
    )


if __name__ == "__main__":
    eos_id = 2000
    pad_id = 2001
    samples = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 2
    steps = 500
    print_interval = 10
    save_interval = 50
    r = 2
    alpha = 8
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
    random.shuffle(texts)
    split = 4
    train_texts, test_texts = texts[:split], texts[split:]
    tokenizer = RegexTokenizer().load("tiny_story.model")

    checkpoint = torch.load("last_checkpoint_RMSNorm_SwiGLU.pt")
    model_config = checkpoint["model_config"]
    model_state_dict = checkpoint["model_state_dict"]

    model = TinyLanguageModel(**model_config).to(device)
    model.load_state_dict(model_state_dict)
    target_modules = ["heads.qkv"]
    test_lora_model(
        model,
        target_modules,
        tokenizer,
        train_texts,
        test_texts,
        r,
        alpha,
        eos_id,
        pad_id,
        print_interval=10,
        save_interval=50,
        path="RMSNorm_SwiGLU_lora_qkv",
    )
    for name, parameter in model.named_parameters():
        if parameter.requires_grad is True:
            print(name)
