import torch

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

    for prompt_len, response_len in [(3, 2), (2, 1), (5, 4)]:
        prompt_ids = torch.randint(10, 100, (prompt_len,)).tolist()
        response_ids = torch.randint(10, 100, (response_len,)).tolist()

        sample = prepare_sft_sample(prompt_ids, response_ids, eos_id)
        samples.append(sample)
        print("单条样本：", sample)

    batch_x, batch_labels = collate_sft_batch(samples, pad_id)

    print("batch_x：\n", batch_x)
    print("batch_labels：\n", batch_labels)
    print("形状：", batch_x.shape, batch_labels.shape)
