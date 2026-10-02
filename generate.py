from tinylanguagemodel import TinyLanguageModel, get_batch
import torch
from MinBPE import RegexTokenizer

torch.manual_seed(39)
device = "cuda" if torch.cuda.is_available() else "cpu"
best_checkpoint = torch.load("./best_checkpoint_rope.pt", device)
model = TinyLanguageModel(**best_checkpoint["model_config"])
model.load_state_dict(best_checkpoint["model_state_dict"])
model = model.to(device)
tokenizer = RegexTokenizer()
tokenizer.load("./tiny_story.model")
data = torch.load("./val_tensor.pt")

model.eval()
prompt, _ = get_batch(data, 1, model.block_size, device)
print(f"prompt len:{prompt.shape[1]}")


result, logits_1, K, V = model.generate(
    prompt[:, :-1],
    10,
    eos_token=tokenizer.special_tokens["<|endoftext|>"],
    top_p=0.8,
)
print(f"result seq_len:{result.shape[1]}")
result, logits_2, K, V = model.generate(
    prompt,
    10,
    K=K,
    V=V,
    eos_token=tokenizer.special_tokens["<|endoftext|>"],
    top_p=0.8,
)
print(f"result seq_len:{result.shape[1]}")
