from MinBPE import RegexTokenizer
from tinylanguagemodel import TinyLanguageModel, train, estimate_loss, get_batch
import time, torch

torch.manual_seed(39)

with open("./train_text.txt", "r", encoding="utf-8") as f:
    train_text = f.read()
with open("./val_text.txt", "r", encoding="utf-8") as f:
    val_text = f.read()


batch_size = 16
block_size = 128
n_embd = 128
num_head = 8
num_layer = 6
steps = 10000
eval_iters = 100
lr = 0.001
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = RegexTokenizer()
tokenizer.load("./tiny_story.model")
vocab_size = len(tokenizer.vocab)

train_data = tokenizer.encode(train_text, {"<|endoftext|>"})
val_data = tokenizer.encode(val_text, {"<|endoftext|>"})
train_data = torch.tensor(train_data, dtype=torch.long)
val_data = torch.tensor(val_data, dtype=torch.long)

model = TinyLanguageModel(
    vocab_size, n_embd, block_size, num_head, num_layer, dropout=0.1
)
model = model.to(device)
optimizer = torch.optim.AdamW(params=model.parameters(), lr=5e-4)
train(
    model=model,
    optimizer=optimizer,
    train_data=train_data,
    val_data=val_data,
    batch_size=batch_size,
    block_size=block_size,
    steps=steps,
    eval_iters=eval_iters,
    device=device,
)
