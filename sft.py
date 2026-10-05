import torch


def prepare_sft_sample(prompt_ids, response_ids, eos_id):
    seq = torch.tensor(data=prompt_ids + response_ids + [eos_id], dtype=torch.long)
    x, label = seq[:-1], seq[1:].clone()
    label[: len(prompt_ids) - 1] = -100
    return x, label


if __name__ == "__main__":
    print(prepare_sft_sample([10, 20, 30], [40, 50], 2000))
