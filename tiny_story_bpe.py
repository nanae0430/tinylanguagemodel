from MinBPE import RegexTokenizer
import time

tokenizer = RegexTokenizer()
with open("./TinyStories-valid.txt", "r", encoding="utf-8") as f:
    text = f.read()
eos_token = "<|endoftext|>"
raw_split_index = int(0.9 * len(text))
eos_index = text.rfind(eos_token, 0, raw_split_index)
train_text = text[0 : eos_index + len(eos_token)]
val_text = text[eos_index + len(eos_token) :]

start_time = time.time()
special_tokens = {"<|endoftext|>": 2000}
tokenizer.register_special_tokens(special_tokens)
tokenizer.train(corpus=train_text, vocab_size=2000)
tokenizer.save("tiny_story")
end_time = time.time()
print(f"{(end_time - start_time):.3f}s")
with open("./train_text.txt", "w", encoding="utf-8") as f:
    f.write(train_text)
with open("./val_text.txt", "w", encoding="utf-8") as f:
    f.write(val_text)
