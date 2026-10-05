import torch


def prepare_sft_sample(prompt_ids, response_ids, eos_id):
    seq = torch.cat([prompt_ids, response_ids, eos_id], dim=-1)
    x, label = seq[:-1], seq[1:]
    label[: len(prompt_ids) - 1] = -100
    return x, label
