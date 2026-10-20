docs = [
    "lora reduces fine tuning cost by freezing pretrained weights and learning a low rank update with small matrices",
    "rope encodes token position by rotating query and key vectors before attention is computed",
    "kv cache stores previously computed keys and values so autoregressive decoding avoids repeated calculations",
    "rmsnorm scales hidden states using the root mean square without subtracting the feature mean",
    "swiglu uses a gated feedforward network with silu activation and separate up gate and down projections",
    "bpe tokenization builds a vocabulary by repeatedly merging frequent adjacent symbol pairs",
    "gradient clipping limits the size of gradients to help stabilize parameter updates during training",
    "learning rate warmup gradually increases the optimizer step size at the beginning of training before decay",
]

queries = [
    "how does lora reduce fine tuning cost",
    "why can pretrained weights stay frozen while two small matrices learn",
    "how does rope modify query and key vectors",
    "how can rotations represent word order in attention",
    "what does kv cache store during decoding",
    "how can text generation reuse calculations from earlier steps",
    "why does rmsnorm not subtract the mean",
    "how can hidden features be normalized using their squared magnitude",
    "what activation does swiglu use",
    "why does a feedforward layer combine gating and projection branches",
    "how does bpe build its vocabulary",
    "how are frequently neighboring symbols combined into larger units",
    "why do we use gradient clipping",
    "how can unusually large training updates be controlled",
    "what is learning rate warmup",
    "why should optimization begin with small steps that gradually grow",
]

labels = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]
