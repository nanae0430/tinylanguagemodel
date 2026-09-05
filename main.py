from MinBPE import RegexTokenizer
from tinylanguagemodel import TinyLanguageModel, train, estimate_loss
import time, torch

torch.manual_seed(39)
vocab_size = 1000
batch_size = 16
block_size = 128
n_embd = 128
num_head = 8
num_layer = 6
steps = 10000
eval_iters = 100
lr = 0.001
device = "cuda" if torch.cuda.is_available() else "cpu"
# with open("./tiny_shakespeare.txt", "r", encoding="utf-8") as f:
#     corpus = f.read()
# start_time = time.time()
# tokenizer = RegexTokenizer()
# tokenizer.train(
#     text=corpus,
#     vocab_size=1000,
# )
# tokenizer.save("tiny_1")
# end_time = time.time()
# print((end_time - start_time), "s")
# tokenizer = RegexTokenizer()
# tokenizer.load("./tiny_1.model")
# ids = tokenizer.encode(text=corpus)
# ids_trained = torch.tensor(
#     data=ids,
#     dtype=torch.long,
#     device="cpu",
# )
# torch.save(ids_tensor, "tiny_token.pt")
# ids = torch.load("./tiny_token.pt")
# ids = ids.to("cpu")
# train_ids, val_ids = ids[: int(len(ids) * 0.9)], ids[int(len(ids) * 0.9) :]
# model = TinyLanguageModel(
#     vocab_size=vocab_size,
#     n_embd=n_embd,
#     block_size=block_size,
#     num_head=num_head,
#     num_layer=num_layer,
#     dropout=0.1,
# )
# model.to(device)
# optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr)
# train(
#     model=model,
#     optimizer=optimizer,
#     train_data=train_ids,
#     val_data=val_ids,
#     batch_size=batch_size,
#     block_size=block_size,
#     steps=steps,
#     eval_iters=eval_iters,
#     device=device,
# )
model = TinyLanguageModel(
    vocab_size=vocab_size,
    n_embd=n_embd,
    block_size=block_size,
    num_head=num_head,
    num_layer=num_layer,
)
best_checkpoint = torch.load("./best_checkpoint.pt")
model.load_state_dict(best_checkpoint["model_state_dict"])
model.to(device)
model.eval()

tokenizer = RegexTokenizer()
tokenizer.load("./tiny_1.model")
idx = tokenizer.encode("I'm Miku.")
idx = torch.tensor(data=idx, dtype=torch.long, device=device).unsqueeze(0)
# 贪心生成：多次运行结果相同
output = model.generate(idx, 100, top_k=1)[0].tolist()
print(tokenizer.decode(output))
print("-" * 20)
# Top-K采样
output = model.generate(idx, 100, top_k=20, temperature=0.8)[0].tolist()
print(tokenizer.decode(output))
print("-" * 20)
# 全词表采样
output = model.generate(idx, 100, top_k=None, temperature=0.8)[0].tolist()
print(tokenizer.decode(output))
print("-" * 20)
