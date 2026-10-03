import torch, time
from tinylanguagemodel import TinyLanguageModel, train, get_batch


def benchmark_training(
    model,
    optimizer,
    data,
    batch_size,
    device,
    warmup_steps=20,
    measure_steps=200,
    use_amp=False,
):
    model.train()
    for _ in range(warmup_steps):
        input_data, target_data = get_batch(
            data=data, batch_size=batch_size, block_size=model.block_size, device=device
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
            data=data, batch_size=batch_size, block_size=model.block_size, device=device
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
    token_num = measure_steps * batch_size * model.block_size
    tokens_per_second = token_num / elapsed_seconds
    print(
        f"elapsed:\t{elapsed_seconds:.6f}s\nms/step:\t{time_per_step:.6f}ms\ntokens/s:\t{tokens_per_second:.6f}\npeak_memory:\t{peak_memory:.6f}"
    )


def benchmark_generate(
    model,
    data_in,
    batch_size,
    max_new_token,
    device,
    warmup_steps=20,
    measure_steps=200,
    use_cache=True,
):
    model.eval()

    for _ in range(warmup_steps):

        data, _ = get_batch(
            data=data_in, batch_size=batch_size, block_size=max_new_token, device=device
        )
        K, V = None, None
        result, logits, K, V = model.generate(
            data, max_new_token, top_k=1, use_cache=use_cache, eos_token=-1
        )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start_time = time.perf_counter()
    K, V = None, None
    for _ in range(measure_steps):
        data, _ = get_batch(
            data=data_in, batch_size=batch_size, block_size=max_new_token, device=device
        )
        K, V = None, None
        result, logits, K, V = model.generate(
            data, max_new_token, top_k=1, use_cache=use_cache, K=K, V=V, eos_token=-1
        )
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    peak_memory = torch.cuda.max_memory_allocated() / 1024**2
    elapsed_seconds = end_time - start_time
    time_per_step = elapsed_seconds / measure_steps * 1000
    token_num = measure_steps * batch_size * max_new_token
    tokens_per_second = token_num / elapsed_seconds
    print(
        f"elapsed:\t{elapsed_seconds:.6f}s\nms/step:\t{time_per_step:.6f}ms\ntokens/s:\t{tokens_per_second:.6f}\npeak_memory:\t{peak_memory:.6f}M"
    )


if __name__ == "__main__":
    # from MinBPE import RegexTokenizer

    # # with open("./train_text.txt", "r", encoding="utf-8") as f:
    # #     train_text = f.read()

    batch_size = 32
    # block_size = 128
    # n_embd = 128
    # num_head = 8
    # num_layer = 8
    # steps = 200
    # eval_iters = 100
    # lr = 0.001
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # tokenizer = RegexTokenizer()
    # tokenizer.load("./tiny_story.model")
    # vocab_size = len(tokenizer.vocab)

    # train_data = tokenizer.encode(train_text, {"<|endoftext|>"})
    # train_data = torch.tensor(train_data, dtype=torch.long)
    # torch.save(train_data, "./train_tensor.pt")
    for path in ["RMSNorm", "LayerNorm"]:
        print('*'*10+path+'*'*10)
        train_data = torch.load("./train_tensor.pt", weights_only=True)
        check_point = torch.load(f"./last_checkpoint_{path}.pt", device)
        config = dict(check_point["model_config"])
        config["norm_type"] = path
        model = TinyLanguageModel(**config).to(device)
        model.load_state_dict(check_point["model_state_dict"])
        optimizer = torch.optim.AdamW(params=model.parameters())
        optimizer.load_state_dict(check_point["optimizer_state_dict"])
        start_step = check_point["step"]
        non_improve = check_point["non_improve"]
        best_val_loss = check_point["best_val_loss"]
        benchmark_training(model, optimizer, train_data, batch_size, device)

    # for i in range(3):
    #     print("*" * 10, f"batch_size={32*2**i}", "*" * 10)
    #     benchmark_training(
    #         model=model,
    #         optimizer=optimizer,
    #         data=train_data,
    #         batch_size=32 * 2**i,
    #         block_size=block_size,
    #         device=device,
    #         use_amp=True,
    #     )
    #     print("-" * 20)
    #     benchmark_training(
    #         model=model,
    #         optimizer=optimizer,
    #         data=train_data,
    #         batch_size=32 * 2**i,
    #         block_size=block_size,
    #         device=device,
    #         use_amp=False,
    #     )
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

    # print("*" * 10 + "use_cache=True" + "*" * 10)
    # benchmark_generate(
    #     model,
    #     train_data,
    #     batch_size=16,
    #     max_new_token=32,
    #     use_cache=True,
    #     device=device,
    # )
    # print("*" * 10 + "use_cache=False" + "*" * 10)
    # benchmark_generate(
    #     model,
    #     train_data,
    #     batch_size=16,
    #     max_new_token=32,
    #     use_cache=False,
    #     device=device,
    # )
    # # # 提前准备输入，两种模式使用同一份数据
    # # data, _ = get_batch(train_data, batch_size, 32, device)
    # # model.eval()
    # # for cache_enabled in (
    # #     False,
    # #     True,
    # #     False,
    # #     True,
    # # ):
    # #     torch.cuda.synchronize()

    # #     with torch.profiler.profile(
    # #         activities=[
    # #             torch.profiler.ProfilerActivity.CPU,
    # #             torch.profiler.ProfilerActivity.CUDA,
    # #         ],
    # #     ) as prof:
    # #         model.generate(
    # #             data,
    # #             32,
    # #             eos_token=-1,
    # #             top_k=1,
    # #             use_cache=cache_enabled,
    # #         )
    # #         torch.cuda.synchronize()

    # #     print(f"use_cache={cache_enabled}")
    # #     print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
    # #     print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
