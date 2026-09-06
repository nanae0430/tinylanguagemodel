import regex, unicodedata


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


def render_token(token):
    text = token.decode(encoding="utf-8", errors="replace")
    result = []

    for ch in text:
        category = unicodedata.category(ch)

        if category.startswith("C"):
            result.append(rf"\u{ord(ch):04x}")
        else:
            result.append(ch)

    return "".join(result)


class Tokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.pattern = ""
        self.special_tokens = {}
        self.special_tokens_inversed = {}

    def train(self, text, vocab_size):
        raise NotImplementedError

    def encode(self, text):
        raise NotImplementedError

    def decode(self, ids):
        token = [self.vocab[id] for id in ids]
        return b"".join(token).decode("utf-8", errors="replace")

    def _encode_chunk(self, ids):
        while len(ids) > 1:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            new_id = self.merges[pair]
            ids = merge(ids, pair, new_id)
        return ids

    def _build_vocab(self):
        vocab = {i: bytes([i]) for i in range(256)}
        for pair, idx in sorted(self.merges.items(), key=lambda item: item[1]):
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
        for token, idx in self.special_tokens.items():
            vocab[idx] = token.encode("utf-8")
        return vocab

    def save(self, file_prefix):
        model_file = file_prefix + ".model"
        vocab_file = file_prefix + ".vocab"
        with open(model_file, "w", encoding="utf-8") as f:
            f.write("minbpe v1\n")
            f.write(f"{self.__class__.__name__}\n")
            f.write(f"{self.pattern}\n")
            f.write(f"{len(self.merges)}\n")
            for pair, idx in sorted(self.merges.items(), key=lambda item: item[1]):
                f.write(f"{pair[0]}\t{pair[1]}\t{idx}\n")
            f.write(f"{len(self.special_tokens)}\n")
            for token, idx in sorted(
                self.special_tokens.items(), key=lambda item: len(item[0]), reverse=True
            ):
                f.write(f"{token}\t{idx}\n")
        with open(vocab_file, "w", encoding="utf-8") as f:
            inverted_merges = {v: k for k, v in self.merges.items()}
            for idx, token in sorted(self.vocab.items(), key=lambda item: item[0]):
                if idx in inverted_merges:
                    left, right = map(
                        lambda parent_idx: render_token(self.vocab[parent_idx]),
                        inverted_merges[idx],
                    )

                    result = f"[{left}]\t[{right}]\t->[{render_token(token)}]\t{idx}\n"
                    f.write(result)
                else:
                    f.write(f"[{render_token(token)}]\t{idx}\n")

    def load(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            version = f.readline().rstrip("\n")
            tokenizer_type = f.readline().rstrip("\n")
            pattern = f.readline().rstrip("\n")
            merge_count = int(f.readline().rstrip("\n"))
            if version != "minbpe v1":
                raise ValueError("不支持的模型格式")
            if tokenizer_type != self.__class__.__name__:
                raise ValueError("不支持的模型格式")
            merges = {}
            for i in range(merge_count):
                parts = f.readline().rstrip("\n").split()
                left, right, idx = map(int, parts)
                merges[(left, right)] = idx
            special_tokens = {}
            special_tokens_count = int(f.readline().rstrip("\n"))
            for i in range(special_tokens_count):
                parts = f.readline().rstrip("\n").rsplit("\t", maxsplit=1)
                token, idx = parts[0], int(parts[1])
                special_tokens[token] = idx
            self.merges = merges
            self.pattern = pattern
            self.special_tokens = special_tokens
            self.special_tokens_inversed = {
                v: k for k, v in self.special_tokens.items()
            }
            self.vocab = self._build_vocab()


class BasicTokenizer(Tokenizer):

    def train(self, text, vocab_size=266):
        if vocab_size < 256:
            raise ValueError("vocab_size不能小于256")
        ids = list(text.encode("utf-8"))
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

    def encode(self, text):
        if not text:
            return []
        ids = list(text.encode("utf-8"))
        ids = self._encode_chunk(ids)
        return ids


class RegexTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()
        self.special_pattern = ""
        self.pattern = (
            r"'(?i:[sdmt]|ll|ve|re)"
            r"|[^\r\n\p{L}\p{N}]?+\p{L}++"
            r"|\p{N}{1,3}+"
            r"| ?[^\s\p{L}\p{N}]++[\r\n]*+"
            r"|\s++$"
            r"|\s*[\r\n]"
            r"|\s+(?!\S)"
            r"|\s"
        )

    def train(self, text, vocab_size=266):
        if vocab_size < 256:
            raise ValueError("vocab_size不能小于256")

        new_id = 256
        merges = {}
        vocab = {i: bytes([i]) for i in range(256)}

        chunks = regex.findall(self.pattern, text)
        chunk_ids = [list(chunk.encode("utf-8")) for chunk in chunks]

        while new_id < vocab_size:
            all_stats = {}

            for ids in chunk_ids:
                stats = get_stats(ids)

                for pair, freq in stats.items():
                    all_stats[pair] = all_stats.get(pair, 0) + freq
            if len(all_stats) <= 0:
                break
            max_pair = max(all_stats, key=all_stats.get)
            chunk_ids = [merge(ids, max_pair, new_id) for ids in chunk_ids]
            merges[max_pair] = new_id
            vocab[new_id] = vocab[max_pair[0]] + vocab[max_pair[1]]
            new_id += 1

        merge_ids = set(merges.values())
        special_ids = set(self.special_tokens.values())
        conflict = merge_ids & special_ids
        if conflict:
            raise ValueError(f"合并词表与特殊token表存在冲突:{conflict}")

        self.merges = merges
        self.vocab = self._build_vocab()

    def train_multitexts(self, corpus, vocab_size=1027):
        if vocab_size < 256:
            raise ValueError("vocab_size不能小于256")
        new_id = 256
        merges = {}
        vocab = {i: bytes([i]) for i in range(256)}

        texts = self._split_special_tokens(corpus, self.special_tokens)
        text_ids = []
        for text in texts:
            if text in self.special_tokens:
                continue
            chunks = regex.findall(self.pattern, text)
            chunk_ids = [list(chunk.encode("utf-8")) for chunk in chunks]
            text_ids.extend(chunk_ids)
        while new_id < vocab_size:
            all_stats = {}
            for ids in text_ids:
                stats = get_stats(ids)

                for pair, freq in stats.items():
                    all_stats[pair] = all_stats.get(pair, 0) + freq
            if len(all_stats) <= 0:
                break
            max_pair = max(all_stats, key=all_stats.get)
            text_ids = [
                [merge(ids, max_pair, new_id) for ids in text_id]
                for text_id in text_ids
            ]
            merges[max_pair] = new_id
            vocab[new_id] = vocab[max_pair[0]] + vocab[max_pair[1]]
            new_id += 1

        merge_ids = set(merges.values())
        special_ids = set(self.special_tokens.values())
        conflict = merge_ids & special_ids
        if conflict:
            raise ValueError(f"合并词表与特殊token表存在冲突:{conflict}")

        self.merges = merges
        self.vocab = self._build_vocab()

    def encode(self, text, allow_special="none"):
        result = []
        if not text:
            return result
        if allow_special == "none":
            active_special = {}

        elif allow_special in ("all", "none_raise"):
            active_special = self.special_tokens

        elif isinstance(allow_special, set):
            active_special = {
                token: idx
                for token, idx in self.special_tokens.items()
                if token in allow_special
            }

        else:
            raise ValueError(f"无法识别allow_special:{allow_special}")
        tokens = self._split_special_tokens(text, active_special)
        for token in tokens:
            if token not in active_special:
                result.extend(self.encode_ordinary(token))
            elif allow_special == "none_raise":
                raise ValueError(f"文本中包含不允许的特殊字符:{token}")
            else:
                result.append(active_special[token])
        return result

    def register_special_tokens(self, special_tokens):
        for token, idx in special_tokens.items():
            if (
                token
                and type(token) is str
                and type(idx) is int
                and idx not in self.vocab
                and idx not in self.special_tokens_inversed
                and token not in self.special_tokens
            ):
                self.special_tokens[token] = idx
                self.special_tokens_inversed[idx] = token
        self.vocab = self._build_vocab()

    def _split_special_tokens(self, text, special_tokens):
        if not special_tokens:
            return [text]

        special_pattern = "|".join(
            regex.escape(token)
            for token in sorted(
                special_tokens,
                key=len,
                reverse=True,
            )
        )
        tokens = regex.split(f"({special_pattern})", text)
        self.special_pattern = special_pattern
        return tokens

    def encode_ordinary(self, text):
        result = []
        chunks = regex.findall(self.pattern, text)
        chunk_ids = [list(chunk.encode("utf-8")) for chunk in chunks]
        chunk_ids = [self._encode_chunk(ids) for ids in chunk_ids]
        for ids in chunk_ids:
            result.extend(ids)
        return result


if __name__ == "__main__":
    tokenizer = RegexTokenizer()
    tokenizer.register_special_tokens({"<|endoftext|>": 2000})
