from tinylanguagemodel import TinyLanguageModel, get_batch
import torch
from MinBPE import RegexTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
best_checkpoint = torch.load("./best_checkpoint_rope.pt", device)
model = TinyLanguageModel(**best_checkpoint["model_config"])
model.load_state_dict(best_checkpoint["model_state_dict"])
model = model.to(device)
tokenizer = RegexTokenizer()
tokenizer.load("./tiny_story.model")
data = torch.load("./val_tensor.pt")

model.eval()
prompt, _ = get_batch(data, 1, 10, device)
for i in range(10):
    result, logits_1 = model.generate(
        prompt,
        1,
        eos_token=tokenizer.special_tokens["<|endoftext|>"],
        top_p=0.8,
    )
    result, _ = model.generate(
        prompt[:, :-1],
        1,
        eos_token=tokenizer.special_tokens["<|endoftext|>"],
        top_p=0.8,
    )
    result, logits_2 = model.generate(
        prompt[:, -1:],
        1,
        K=model.K,
        V=model.V,
        eos_token=tokenizer.special_tokens["<|endoftext|>"],
        top_p=0.8,
    )
    prompt = result
    print(f"{i+1}:\t", (logits_1 - logits_2).abs().max().item())
