from tinylanguagemodel import TinyLanguageModel
import torch
from MinBPE import RegexTokenizer

batch_size = 16
block_size = 128
n_embd = 128
num_head = 8
num_layer = 6
steps = 10000
eval_iters = 100
lr = 0.001
device = "cuda" if torch.cuda.is_available() else "cpu"
vocab_size = 2001
best_checkpoint = torch.load("./best_checkpoint.pt", device)
model = TinyLanguageModel(
    vocab_size=vocab_size,
    n_embd=n_embd,
    block_size=block_size,
    num_head=num_head,
    num_layer=num_layer,
    dropout=0.2,
)
model.load_state_dict(best_checkpoint["model_state_dict"])
model = model.to(device)
tokenizer = RegexTokenizer()
tokenizer.load("./tiny_story.model")
prompt = "once upon a time"
prompt = tokenizer.encode(prompt)
prompt = torch.tensor(prompt, dtype=torch.long, device=device).unsqueeze(0)
prompt = prompt.repeat(4, 1)
model.eval()
result = model.generate(
    prompt, 1000, eos_token=tokenizer.special_tokens["<|endoftext|>"], top_k=50
)
eos_token = tokenizer.special_tokens["<|endoftext|>"]
for response in result:

    response = response.tolist()

    if eos_token in response:
        response = response[: 1 + response.index(eos_token)]
    response = tokenizer.decode(response)
    print(response)
    print("*" * 20)
