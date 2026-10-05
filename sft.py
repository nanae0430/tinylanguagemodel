import torch
from tinylanguagemodel import TinyLanguageModel

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


if __name__ == "__main__":
    eos_id = 2000
    pad_id = 2001
    samples = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for prompt_len, response_len in [(3, 2), (2, 1), (5, 4)]:
        prompt_ids = torch.randint(10, 100, (prompt_len,)).tolist()
        response_ids = torch.randint(10, 100, (response_len,)).tolist()

        sample = prepare_sft_sample(prompt_ids, response_ids, eos_id)
        samples.append(sample)
        print("单条样本：", sample)

    batch_x, batch_labels = collate_sft_batch(samples, pad_id)

    checkpoint = torch.load("last_checkpoint_RMSNorm_SwiGLU.pt")
    model_config = checkpoint["model_config"]
    model_state_dict = checkpoint["model_state_dict"]
    optimizer_state_dict = checkpoint["optimizer_state_dict"]
    model = TinyLanguageModel(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters())
    optimizer.load_state_dict(optimizer_state_dict)
    batch_x, batch_labels = batch_x.to(device), batch_labels.to(device)
    logits, loss = model(batch_x, batch_labels)
    optimizer.zero_grad()
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"{name}梯度为 None"
        assert torch.isfinite(parameter.grad).all().item(), f"{name}梯度为NaN或inf"
    print("梯度检查通过")
