from MinBPE import RegexTokenizer

tokenizer = RegexTokenizer()
with open("./TinyStories-valid.txt", "r", encoding="utf-8") as f:
    text = f.read()

special_tokens = {"<|endoftext|>": 2000}
tokenizer.register_special_tokens(special_tokens)
tokenizer.train_multitexts(corpus=text, vocab_size=2000)
tokenizer.save("tiny_story")
