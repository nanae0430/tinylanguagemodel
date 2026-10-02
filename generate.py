from tinylanguagemodel import TinyLanguageModel
import torch
from MinBPE import RegexTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
best_checkpoint = torch.load("./best_checkpoint_rope.pt", device)
model = TinyLanguageModel(**best_checkpoint["model_config"])
model.load_state_dict(best_checkpoint["model_state_dict"])
model = model.to(device)
tokenizer = RegexTokenizer()
tokenizer.load("./tiny_story.model")
prompt = "once upon a time"
prompt = tokenizer.encode(prompt)
prompt = torch.tensor(prompt, dtype=torch.long, device=device).unsqueeze(0)

model.eval()
for i in range(10):
    result, logits_1 = model.generate(
        prompt,
        1,
        eos_token=tokenizer.special_tokens["<|endoftext|>"],
        top_p=0.8,
    )
    result, logits_2 = model.generate(
        prompt[:, :-1],
        2,
        eos_token=tokenizer.special_tokens["<|endoftext|>"],
        top_p=0.8,
    )
    print("*" * 10 + str(i + 1) + "*" * 10)
    print(max(logits_1 - logits_2))
    prompt = result
