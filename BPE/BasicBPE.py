def get_stats(ids):
    frequency = {}
    pairs = zip(ids, ids[1:])
    for pair in pairs:
        if pair not in frequency:
            frequency[pair] = 1
        else:
            frequency[pair] += 1

    return frequency


def merge(ids, pair, new_id):
    i, n = 0, len(ids)
    new_ids = []
    while i < n - 1:
        if ids[i] != pair[0] or ids[i + 1] != pair[1]:
            new_ids.append(ids[i])
            i += 1
        else:
            new_ids.append(new_id)
            i += 2
    if i == n - 1:
        new_ids.append(ids[-1])
    return new_ids



class BasicTokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

    def train(self, corpus, vocab_size=266):
        if vocab_size < 256:
            raise ValueError("vocab_size不能小于256")
        ids = list(corpus.encode())
        new_id = 256
        merges = {}
        vocab = {i: bytes([i]) for i in range(256)}
        while new_id < vocab_size:
            stats = get_stats(ids)
            if len(stats) <= 0:
                break
            max_pair = max(stats, key=stats.get)
            ids = merge(ids, max_pair, new_id)
            merges[max_pair] = new_id
            vocab[new_id] = vocab[max_pair[0]] + vocab[max_pair[1]]
            new_id += 1
        self.merges = merges
        self.vocab = vocab
        return ids

    def encode(self, text):
        if not text:
            return []
        ids = list(text.encode())
        while len(ids) > 1:
            stats = get_stats(ids)
            pair = min(
                stats,
                key=lambda p: self.merges.get(p, float("inf")),
            )
            if pair not in self.merges:
                break
            new_id = self.merges[pair]
            ids = merge(ids, pair, new_id)
        return ids

    def decode(self, ids):
        token = [self.vocab[id] for id in ids]
        return b"".join(token).decode("utf-8", errors="replace")


def test_roundtrip():
    tokenizer = BasicTokenizer()

    tokenizer.train("ababab", 257)
    assert tokenizer.vocab[256] == b"ab"
    assert tokenizer.decode(tokenizer.encode("ab")) == "ab"
    tokenizer.train("cdcdcd", 257)
    assert tokenizer.vocab[256] == b"cd"

    train_data = [
        "",
        "a",
        "hello world",
        "你好，世界",
        "hello你好🙂",
    ]
    for i in train_data:
        tokenizer.train(i)
        assert tokenizer.decode(tokenizer.encode(i)) == i


if __name__ == "__main__":
    # test_roundtrip()
    import regex

    text = "hello123!!! 你好42"

    pattern = r"\p{L}+|\p{N}+|\s+|[^\p{L}\p{N}\s]+"

    chunks = regex.findall(pattern, text)

    print(chunks)
