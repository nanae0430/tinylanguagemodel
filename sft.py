import torch, re, random, json
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


def get_sft_batch(
    texts,
    tokenizer,
    batch_size,
    eos_id,
    pad_id,
    use_samples=None,
):
    if use_samples is not None:
        batch_sample = random.sample(samples, k=batch_size)
        return collate_sft_batch(batch_sample, pad_id)

    batch_texts = random.sample(texts, k=batch_size)
    return prepare_sft_text_sample(batch_texts, tokenizer, eos_id, pad_id)


def get_sft_batch_sample(
    samples: list[tuple[torch.Tensor, torch.Tensor]], batch_size, pad_id: int
):
    batch_sample = random.sample(samples, k=batch_size)
    return collate_sft_batch(batch_sample, pad_id)


def eval_for_sft(
    model,
    train_texts,
    test_texts,
    tokenizer,
    batch_size,
    device,
    eos_id,
    pad_id,
    use_samples_train=None,
    use_samples_val=None,
    eval_iters=1,
):
    model.eval()
    train_losses, eval_losses = [], []

    with torch.no_grad():
        for _ in range(eval_iters):
            test_question, test_answer = get_sft_batch(
                test_texts, tokenizer, batch_size, eos_id, pad_id, use_samples_val
            )
            train_question, train_answer = get_sft_batch(
                train_texts, tokenizer, batch_size, eos_id, pad_id, use_samples_train
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
    lora_config,
    use_samples_train=None,
    use_samples_val=None,
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
            train_texts,
            tokenizer,
            batch_size,
            eos_id,
            pad_id,
            use_samples=use_samples_train,
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
                use_samples_train=use_samples_train,
                use_samples_val=use_samples_val,
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
                lora_config=lora_config,
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


def merge_lora(model: TinyLanguageModel, target_modules, file_path):
    paths = [target_module.split(".") for target_module in target_modules]
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    for block in model.model[:-1]:
        for path in paths:
            parent = block
            for p in path[:-1]:
                parent = getattr(parent, p)
            lora_layer: LoRALinear = getattr(parent, path[-1])
            w_merged = (
                lora_layer.base_layer.weight
                + lora_layer.scaling * lora_layer.B.weight @ lora_layer.A.weight
            )
            layer_merged = torch.nn.Linear(
                lora_layer.base_layer.in_features,
                lora_layer.base_layer.out_features,
                bias=lora_layer.base_layer.bias is not None,
                device=device,
                dtype=dtype,
            )
            with torch.no_grad():
                layer_merged.weight.copy_(w_merged)
                if lora_layer.base_layer.bias is not None:
                    layer_merged.bias.copy_(lora_layer.base_layer.bias)
            setattr(parent, path[-1], layer_merged)
    torch.save(
        {
            "model_config": model.model_config,
            "model_state_dict": model.state_dict(),
        },
        f"{file_path}.pt",
    )


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
    lora_config = {"r": r, "alpha": alpha, "target_modules": target_modules}
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
        lora_config,
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


def load_lora(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path)
    step = checkpoint["step"]
    model_config = checkpoint["model_config"]
    model_state_dict = checkpoint["model_state_dict"]
    optimizer_state_dict = checkpoint["optimizer_state_dict"]
    train_loss = checkpoint["train_loss"]
    val_loss = checkpoint["val_loss"]
    best_val_loss = checkpoint["best_val_loss"]
    non_improve = checkpoint["non_improve"]

    model = TinyLanguageModel(**model_config).to(device)
    model.requires_grad_(False)
    lora_config = checkpoint["lora_config"]
    apply_lora(model=model, **lora_config)
    model.load_state_dict(model_state_dict)
    optim_param = [
        parameter for parameter in model.parameters() if parameter.requires_grad is True
    ]
    optimizer = torch.optim.AdamW(optim_param)
    optimizer.load_state_dict(optimizer_state_dict)

    return model, optimizer, step, train_loss, val_loss, best_val_loss, non_improve


if __name__ == "__main__":
    # eos_id = 2000
    # pad_id = 2001
    # samples = []
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # batch_size = 2
    # steps = 500
    # print_interval = 10
    # save_interval = 50
    # r = 2
    # alpha = 8
    # texts = """User: What color is the sky on a sunny day?
    #         Assistant: The sky is blue.
    #         <|endoftext|>
    #         User: What sound does a cat make?
    #         Assistant: A cat says meow.
    #         <|endoftext|>
    #         User: What is one plus two?
    #         Assistant: One plus two is three.
    #         <|endoftext|>
    #         User: Lily feels cold. What can she wear?
    #         Assistant: Lily can wear a warm coat.
    #         <|endoftext|>
    #         User: Tom sees his friend fall down. What should he do?
    #         Assistant: Tom should help his friend get up and ask if they are hurt.
    #         <|endoftext|>
    #         User: Tell me a short story about a dog.
    #         Assistant: A little dog found a red ball in the park. He brought it to his owner, and they played together.
    #         <|endoftext|>"""
    # texts = [part.strip() for part in texts.split("<|endoftext|>") if part.strip()]
    # random.shuffle(texts)
    # split = 4
    # train_texts, test_texts = texts[:split], texts[split:]
    # tokenizer = RegexTokenizer().load("tiny_story.model")

    # # checkpoint = torch.load("last_checkpoint_RMSNorm_SwiGLU.pt")
    # # model_config = checkpoint["model_config"]
    # # model_state_dict = checkpoint["model_state_dict"]

    # # model = TinyLanguageModel(**model_config).to(device)
    # # model.load_state_dict(model_state_dict)
    # target_modules = ["heads.qkv"]
    # # test_lora_model(
    # #     model,
    # #     target_modules,
    # #     tokenizer,
    # #     train_texts,
    # #     test_texts,
    # #     r,
    # #     alpha,
    # #     eos_id,
    # #     pad_id,
    # #     print_interval=10,
    # #     save_interval=50,
    # #     path="RMSNorm_SwiGLU_lora_qkv",
    # # )
    # # model.eval()
    # test_question, _ = get_sft_batch(test_texts, tokenizer, 1, eos_id, pad_id)
    # test_question = test_question.to(device)
    # # logits1, _ = model(test_question)

    # model2, optimizer, step, train_loss, val_loss, best_val_loss, non_improve = (
    #     load_lora("last_checkpoint_RMSNorm_SwiGLU_lora_qkv.pt", device)
    # )
    # model2.eval()
    # with torch.no_grad():
    #     logits2, _ = model2(test_question)

    # merge_lora(model2, target_modules, "merged_RMSNorm_SwiGLU_lora_qkv")
    # model2.eval()
    # with torch.no_grad():
    #     logits3, _ = model2(test_question)
    # assert logits3.shape == logits2.shape
    # assert torch.allclose(logits3, logits2, rtol=1e-5, atol=1e-5)
    # print(f"atol:{1e-5}\nmax abs error:{(logits2 - logits3).abs().max().item():.4e}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = RegexTokenizer().load("tiny_story.model")
    vocab_size = len(tokenizer.vocab)
    checkpoint = torch.load("best_checkpoint_RMSNorm_SwiGLU.pt")
    model_config = checkpoint["model_config"]
    model_state_dict = checkpoint["model_state_dict"]
    model = TinyLanguageModel(**model_config).to(device)
    model.load_state_dict(model_state_dict)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-4)
    block_size = model.block_size
    eos_id = 2000
    pad_id = 2001

    # with open("databricks-dolly-15k.jsonl", "r", encoding="utf-8") as f:
    #     datas = [json.loads(line) for line in f if line.strip()]

    # data_filtered = []
    samples = []

    # for data in datas:
    #     instruction = "User:" + data["instruction"] + "\n"
    #     context = data["context"] + "\n"
    #     response = "Assistant:" + data["response"]
    #     instruction = tokenizer.encode(instruction)
    #     context = tokenizer.encode(context)
    #     response = tokenizer.encode(response)
    #     if 0 < len(instruction + context + response) <= block_size:
    #         data_filtered.append(data)
    #         samples.append(prepare_sft_sample(context + instruction, response, eos_id))
    # print(len(data_filtered))

    # avg_total = sum(x.numel() for x, y in samples) / len(samples)
    # avg_response = sum((y != -100).sum().item() - 1 for x, y in samples) / len(samples)
    # avg_prompt = avg_total - avg_response

    # print(f"平均 Prompt 长度：{avg_prompt:.2f}")
    # print(f"平均 Response 长度：{avg_response:.2f}")
    # print(f"平均总长度：{avg_total:.2f}")

    # with open("data_filtered.jsonl", "w", encoding="utf-8") as f:
    #     for data in data_filtered:
    #         f.write(json.dumps(data, ensure_ascii=False) + "\n")
    with open("data_filtered.jsonl", "r", encoding="utf-8") as f:
        datas = [json.loads(line) for line in f if line.strip()]
    for data in datas:
        instruction = "User:" + data["instruction"] + "\n"
        context = data["context"] + "\n"
        response = "Assistant:" + data["response"]
        instruction = tokenizer.encode(instruction)
        context = tokenizer.encode(context)
        response = tokenizer.encode(response)
        samples.append(prepare_sft_sample(context + instruction, response, eos_id))
    random.shuffle(samples)
    train_data, val_data = (
        samples[: int(0.9 * len(samples))],
        samples[int(0.9 * len(samples)) :],
    )
