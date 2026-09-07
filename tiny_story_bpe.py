from MinBPE import RegexTokenizer
import time

tokenizer = RegexTokenizer()
with open("./TinyStories-valid.txt", "r", encoding="utf-8") as f:
    text = f.read()
start_time = time.time()
special_tokens = {"<|endoftext|>": 2000}
tokenizer.register_special_tokens(special_tokens)
tokenizer.train_include_special_token(corpus=text, vocab_size=2000)
tokenizer.save("tiny_story")
end_time = time.time()
print(f"{(end_time - start_time):.3f}s")
