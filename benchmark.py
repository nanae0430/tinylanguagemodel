import torch, time
from tinylanguagemodel import TinyLanguageModel, train, get_batch


def benchmark_training(
    model,
    optimizer,
    data,
    batch_size,
    block_size,
    device,
    warmup_steps=20,
    measure_steps=200,
    use_amp=False,
):
    model.train()
    for _ in range(warmup_steps):
        input_data, target_data = get_batch(
            data=data, batch_size=batch_size, block_size=block_size, device=device
        )
        optimizer.zero_grad()
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
            logits, loss = model(input_data, target_data)
        loss.backward()
        optimizer.step()
    print(f"logits.dtype:{logits.dtype}\tloss.dtype:{loss.dtype}")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start_time = time.perf_counter()

    for _ in range(measure_steps):
        input_data, target_data = get_batch(
            data=data, batch_size=batch_size, block_size=block_size, device=device
        )
        optimizer.zero_grad()
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
            _, loss = model(input_data, target_data)
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()
    end_time = time.perf_counter()
    peak_memory = torch.cuda.max_memory_allocated() / 1024**2
    elapsed_seconds = end_time - start_time
    time_per_step = elapsed_seconds / measure_steps * 1000
    token_num = measure_steps * batch_size * block_size
    tokens_per_second = token_num / elapsed_seconds
    print(
        f"elapsed:\t{elapsed_seconds:.6f}s\nms/step:\t{time_per_step:.6f}ms\ntokens/s:\t{tokens_per_second:.6f}\npeak_memory:\t{peak_memory:.6f}"
    )


if __name__ == "__main__":
    from MinBPE import RegexTokenizer

    with open("./train_text.txt", "r", encoding="utf-8") as f:
        train_text = f.read()

    batch_size = 32
    block_size = 128
    n_embd = 128
    num_head = 8
    num_layer = 8
    steps = 200
    eval_iters = 100
    lr = 0.001
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # tokenizer = RegexTokenizer()
    # tokenizer.load("./tiny_story.model")
    # vocab_size = len(tokenizer.vocab)

    # train_data = tokenizer.encode(train_text, {"<|endoftext|>"})
    # train_data = torch.tensor(train_data, dtype=torch.long)
    # torch.save(train_data, "./train_tensor.pt")
    train_data = torch.load("./train_tensor.pt")
    check_point = torch.load("./last_checkpoint.pt", device)
    model = TinyLanguageModel(**check_point["model_config"]).to(device)
    model.load_state_dict(check_point["model_state_dict"])
    optimizer = torch.optim.AdamW(params=model.parameters())
    optimizer.load_state_dict(check_point["optimizer_state_dict"])
    start_step = check_point["step"]
    non_improve = check_point["non_improve"]
    best_val_loss = check_point["best_val_loss"]

    for i in range(3):
        print("*" * 10, f"batch_size={32*2**i}", "*" * 10)
        benchmark_training(
            model=model,
            optimizer=optimizer,
            data=train_data,
            batch_size=32 * 2**i,
            block_size=block_size,
            device=device,
            use_amp=True,
        )
        print("-" * 20)
        benchmark_training(
            model=model,
            optimizer=optimizer,
            data=train_data,
            batch_size=32 * 2**i,
            block_size=block_size,
            device=device,
            use_amp=False,
        )
    # grad_norm = torch.empty(steps, device=device)
    # for step in range(steps):
    #     input_data, target_data = get_batch(
    #         data=train_data, batch_size=batch_size, block_size=block_size, device=device
    #     )
    #     optimizer.zero_grad()
    #     with torch.autocast(device_type=device, dtype=torch.bfloat16):
    #         logits, loss = model(input_data, target_data)
    #     loss.backward()
    #     grad_norm[step] = torch.nn.utils.clip_grad_norm_(
    #         model.parameters(), float("inf")
    #     )

    #     optimizer.step()
    # print(f"max grad norm:{torch.max(grad_norm)}")
    # print(f"average grad norm:{torch.mean(grad_norm)}")
    # print(f"min grad norm:{torch.min(grad_norm)}")
