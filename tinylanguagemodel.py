import torch, math
import torch.nn as nn
import torch.nn.functional as F


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(data) <= block_size:
        raise ValueError
    start = torch.randint(0, len(data) - block_size, size=(batch_size, 1))
    offset = torch.arange(0, block_size)

    start_index = start + offset
    input_data = data[start_index]
    target_data = data[start_index + 1]
    input_data = input_data.to(device)
    target_data = target_data.to(device)
    return (input_data, target_data)


class Head(nn.Module):
    def __init__(
        self,
        n_embd: int,
        head_size: int,
        block_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        block_size = self.tril.shape[0]
        B, T, C = x.shape
        if T > block_size:
            raise ValueError("invalid sequence length")
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        head_size = k.shape[-1]
        k_transposed = k.transpose(-1, -2)
        attention_score = q @ k_transposed / head_size**0.5

        mask = self.tril[:T, :T]
        attention_score_masked = torch.masked_fill(
            attention_score,
            mask == 0,
            value=float("-inf"),
        )
        attention_weight = F.softmax(attention_score_masked, dim=-1)
        weight_dropout = self.dropout(attention_weight)
        output = weight_dropout @ v
        return output


class MultiHeadAttention(nn.Module):

    def __init__(self, n_embd, num_head, block_size, dropout=0):
        super().__init__()
        if n_embd % num_head != 0:
            raise ValueError("invalid n_embed and num_head")
        self.heads = nn.ModuleList(
            [
                Head(n_embd, n_embd // num_head, block_size, dropout)
                for _ in range(num_head)
            ]
        )
        self.projection = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        output_seperated = [head(x) for head in self.heads]
        output = torch.concat(output_seperated, dim=-1)
        projection = self.projection(output)
        return self.dropout(projection)


class MultiHeadAttention_v2(nn.Module):
    def __init__(self, n_embd, num_head, block_size, dropout=0.0):
        super().__init__()
        assert n_embd % num_head == 0
        self.num_head = num_head
        self.head_size = n_embd // num_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.attention_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(n_embd, n_embd)
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(1, 1, block_size, block_size)),
            persistent=False,
        )

    def forward(self, x):
        B, T, C = x.shape
        if T > self.tril.shape[-1]:
            raise ValueError(f"序列长度{T}超出上下文长度")
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.num_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.num_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.num_head, self.head_size).transpose(1, 2)
        k_t = k.transpose(-1, -2)
        mask = self.tril[:, :, :T, :T]

        attention_score = torch.masked_fill(
            q @ k_t / self.head_size**0.5, mask == 0, float("-inf")
        )
        attention_weight = F.softmax(attention_score, dim=-1)
        heads_output = self.attention_dropout(attention_weight) @ v

        concat_output = heads_output.transpose(1, 2).contiguous().view(B, T, C)
        projection_output = self.projection(concat_output)
        return self.residual_dropout(projection_output)


class MultiHeadAttentionSDPA(nn.Module):

    def __init__(self, n_embd, num_head, block_size, dropout=0.0):
        super().__init__()
        assert n_embd % num_head == 0
        self.num_head = num_head
        self.head_size = n_embd // num_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.attention_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(n_embd, n_embd)
        self.block_size = block_size

    def forward(self, x):
        B, T, C = x.shape
        if T > self.block_size:
            raise ValueError(f"序列长度{T}超出上下文长度")
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.num_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.num_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.num_head, self.head_size).transpose(1, 2)
        q, k = rope(q, k)
        heads_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.attention_dropout.p if self.training else 0.0,
            is_causal=True,
        )

        concat_output = heads_output.transpose(1, 2).contiguous().view(B, T, C)
        projection_output = self.projection(concat_output)
        return self.residual_dropout(projection_output)


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout=0):
        super().__init__()
        self.feedforward = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.feedforward(x)


class TransformerBlock(nn.Module):

    def __init__(self, n_embd, num_head, block_size, dropout=0):
        super().__init__()
        self.heads = MultiHeadAttentionSDPA(n_embd, num_head, block_size, dropout)
        self.attention_norm = nn.LayerNorm(n_embd)
        self.feedforward_norm = nn.LayerNorm(n_embd)
        self.feedforward = FeedForward(n_embd, dropout)

    def forward(self, x):
        x1 = x + self.heads(self.attention_norm(x))
        y = x1 + self.feedforward(self.feedforward_norm(x1))
        return y


class TinyLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_embd: int,
        block_size: int,
        num_head: int,
        num_layer: int,
        dropout: float = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.block_size = block_size
        self.model_config = {
            "vocab_size": vocab_size,
            "n_embd": n_embd,
            "block_size": block_size,
            "num_layer": num_layer,
            "num_head": num_head,
            "dropout": dropout,
        }
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.model = nn.Sequential(
            *[
                TransformerBlock(n_embd, num_head, block_size, dropout)
                for _ in range(num_layer)
            ],
            nn.LayerNorm(n_embd),
        )

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:

        B, T = idx.shape
        position = torch.arange(T, device=idx.device)
        token_embedding = self.token_embedding_table(idx)
        position_embedding = self.position_embedding_table(position)
        token_embedding = token_embedding + position_embedding
        result = self.model(token_embedding)
        logits = self.lm_head(result)

        if targets is None:
            loss = None
        else:
            result_reshaped = logits.reshape((-1, self.vocab_size))
            targets_reshaped = targets.reshape((-1))
            loss = F.cross_entropy(result_reshaped, targets_reshaped)

        return (logits, loss)

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens,
        eos_token,
        top_k: int = None,
        top_p: float = None,
        temperature: float = 0.8,
    ):

        result = idx

        stop = torch.zeros(size=(idx.shape[0], 1), dtype=torch.bool, device=idx.device)
        if temperature <= 0:
            raise ValueError("temperature必须大于0")

        for _ in range(max_new_tokens):
            context = result[:, max(0, result.shape[1] - self.block_size) :]
            logits, _ = self(context)
            logits = logits[:, -1, :]
            k = logits.shape[-1]

            if top_p is None:
                if top_k == 1:
                    new_token = torch.argmax(logits, dim=-1, keepdim=True)
                elif top_k is not None and top_k <= 0:
                    raise ValueError("top_k必须大于0")
                elif top_k is not None:
                    k = min(top_k, logits.shape[-1])
                    topk_logits, topk_indices = torch.topk(
                        logits / temperature, k, dim=-1
                    )
                    topk_prob = torch.softmax(topk_logits, dim=-1)
                    sample_position = torch.multinomial(topk_prob, 1)
                    new_token = torch.gather(
                        topk_indices, dim=-1, index=sample_position
                    )
                else:
                    prob = torch.softmax(logits / temperature, dim=-1)
                    new_token = torch.multinomial(prob, 1)
            elif top_p <= 0 or top_p > 1:
                raise ValueError("invalid top_p")
            else:
                sorted_prob, sorted_indices = torch.sort(
                    torch.softmax(logits / temperature, dim=-1), dim=-1, descending=True
                )
                cumsum_prob = torch.cumsum(sorted_prob, dim=-1)
                mask = cumsum_prob > top_p
                mask[:, 1:] = mask[:, :-1].clone()
                mask[:, 0] = False
                filter_prob = torch.masked_fill(sorted_prob, mask=mask, value=0)
                filter_prob = filter_prob / torch.sum(filter_prob, dim=-1, keepdim=True)
                sample_position = torch.multinomial(filter_prob, 1)
                new_token = torch.gather(sorted_indices, dim=-1, index=sample_position)
            new_token = new_token.masked_fill(stop, eos_token)
            result = torch.concat((result, new_token), dim=-1)
            stop |= new_token == eos_token
            if stop.all():
                break
        return result


def rope(q, k):
    B, H, T, d = q.shape
    frequency = 10000.0 ** (torch.arange(0, d // 2, device=q.device) / -d * 2)
    frequency = torch.arange(0, T, device=q.device).view(T, 1) * frequency
    sin_vector = torch.sin(frequency)
    cos_vector = torch.cos(frequency)
    even_q, odd_q = (
        q[:, :, :, ::2] * cos_vector - q[:, :, :, 1::2] * sin_vector,
        q[:, :, :, ::2] * sin_vector + q[:, :, :, 1::2] * cos_vector,
    )
    even_q = even_q.view(B, H, T, d // 2, 1)
    odd_q = odd_q.view(B, H, T, d // 2, 1)
    q = torch.concat(tensors=[even_q, odd_q], dim=-1).reshape(B, H, T, d)
    even_k, odd_k = (
        k[:, :, :, ::2] * cos_vector - k[:, :, :, 1::2] * sin_vector,
        k[:, :, :, ::2] * sin_vector + k[:, :, :, 1::2] * cos_vector,
    )
    even_k = even_k.view(B, H, T, d // 2, 1)
    odd_k = odd_k.view(B, H, T, d // 2, 1)
    k = torch.concat(tensors=[even_k, odd_k], dim=-1).reshape(B, H, T, d)
    return q, k


def rope_(q, k):
    B, H, T, d = q.shape
    frequency = 10000.0 ** (torch.arange(0, d // 2, device=q.device) / -d * 2)
    frequency = torch.arange(0, T, device=q.device).view(T, 1) * frequency
    sin_vector = torch.sin(frequency)
    cos_vector = torch.cos(frequency)
    q[:, :, :, ::2], q[:, :, :, 1::2] = (
        q[:, :, :, ::2] * cos_vector - q[:, :, :, 1::2] * sin_vector,
        q[:, :, :, ::2] * sin_vector + q[:, :, :, 1::2] * cos_vector,
    )
    k[:, :, :, ::2], k[:, :, :, 1::2] = (
        k[:, :, :, ::2] * cos_vector - k[:, :, :, 1::2] * sin_vector,
        k[:, :, :, ::2] * sin_vector + k[:, :, :, 1::2] * cos_vector,
    )
    return q, k


def train(
    model,
    optimizer,
    train_data,
    val_data,
    batch_size,
    block_size,
    steps,
    eval_iters,
    device,
    max_lr=1e-4,
    patience=3,
    min_delta=0.01,
    best_val_loss=float("inf"),
    start_step=0,
    non_improve=0,
    is_eval=True,
):

    model.train()

    for step in range(start_step + 1, 1 + steps):
        lr = get_lr(step, int(0.05 * steps), steps, max_lr, 0.1 * max_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        input_x, target = get_batch(
            data=train_data,
            batch_size=batch_size,
            block_size=block_size,
            device=device,
        )
        optimizer.zero_grad()
        _, loss = model(input_x, target)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        if step % 100 == 0:
            print(
                f"{step}\ttrain loss:{loss.item():.6f}\tlr:{lr:.6f}\tgrad norm:{grad_norm:.6f}"
            )
        if (step % 1000 == 0 or step == steps) and is_eval:
            train_loss, eval_loss = estimate_loss(
                model=model,
                train_data=train_data,
                val_data=val_data,
                batch_size=batch_size,
                block_size=block_size,
                eval_iters=eval_iters,
                device=device,
            )
            print(f"{step}\ttrain loss:{train_loss:.6f}\teval loss:{eval_loss:.6f}")
            best_val_loss, non_improve = save_model(
                best_val_loss=best_val_loss,
                current_train_loss=train_loss,
                current_eval_loss=eval_loss,
                min_delta=min_delta,
                model=model,
                non_improve=non_improve,
                optimizer=optimizer,
                step=step,
            )
            if non_improve >= patience:
                return


def save_model(
    best_val_loss,
    current_train_loss,
    current_eval_loss,
    non_improve,
    min_delta,
    model,
    optimizer,
    step,
):
    if current_eval_loss < best_val_loss - min_delta:
        best_val_loss = current_eval_loss
        non_improve = 0
        torch.save(
            {
                "step": step,
                "model_config": model.model_config,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": current_train_loss,
                "val_loss": best_val_loss,
                "best_val_loss": best_val_loss,
                "non_improve": non_improve,
            },
            "best_checkpoint.pt",
        )
    else:
        non_improve += 1
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "model_config": model.model_config,
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": current_train_loss,
            "val_loss": current_eval_loss,
            "best_val_loss": best_val_loss,
            "non_improve": non_improve,
        },
        "last_checkpoint.pt",
    )
    return best_val_loss, non_improve


@torch.no_grad()
def estimate_loss(
    model,
    train_data,
    val_data,
    batch_size,
    block_size,
    eval_iters,
    device,
):

    model.eval()
    train_loss = []
    eval_loss = []
    for i in range(eval_iters):
        input_x, target = get_batch(
            data=train_data,
            batch_size=batch_size,
            block_size=block_size,
            device=device,
        )
        _, loss = model(input_x, target)
        # if i % 10 == 0:
        #     print(i, "train eval:", loss.item())
        train_loss.append(loss.item())
    for i in range(eval_iters):
        input_x, target = get_batch(
            data=val_data,
            batch_size=batch_size,
            block_size=block_size,
            device=device,
        )
        _, loss = model(input_x, target)
        # if i % 100 == 0:
        #     print(i, "val eval:", loss.item())
        eval_loss.append(loss.item())
    model.train()
    return sum(train_loss) / len(train_loss), sum(eval_loss) / len(eval_loss)


def get_lr(step, warmup_step, total_step, max_lr, min_lr):
    if 0 <= step < warmup_step:
        return max_lr * step / warmup_step
    elif step <= total_step:
        return min_lr + (max_lr - min_lr) * 0.5 * (
            1 + math.cos(math.pi * (step - warmup_step) / (total_step - warmup_step))
        )
    else:
        return min_lr


if __name__ == "__main__":
    #     device = "cuda" if torch.cuda.is_available() else "cpu"
    #     batch_size = 4
    #     vocab_size = 100
    #     block_size = 16
    #     n_embd = 128
    #     num_head = 8
    #     num_layer = 12
    #     dropout = 0.1
    #     steps = 100
    #     data = torch.randint(
    #         0,
    #         100,
    #         size=(1000,),
    #         dtype=torch.long,
    #         device=device,
    #     )
    #     train_data, val_data = data[:900], data[900:]
    #     model = TinyLanguageModel(
    #         vocab_size=vocab_size,
    #         n_embd=n_embd,
    #         block_size=block_size,
    #         num_head=num_head,
    #         num_layer=num_layer,
    #         dropout=dropout,
    #     ).to(device)
    #     optimizer = torch.optim.AdamW(
    #         params=model.parameters(),
    #         lr=1e-3,
    #     )
    #     model, optimizer = train(
    #         model=model,
    #         optimizer=optimizer,
    #         data=train_data,
    #         batch_size=batch_size,
    #         block_size=block_size,
    #         steps=steps,
    #         device=device,
    #     )
    #     train_loss, eval_loss = estimate_loss(
    #         model=model,
    #         train_data=train_data,
    #         val_data=val_data,
    #         batch_size=batch_size,
    #         block_size=block_size,
    #         eval_iters=20,
    #         device=device,
    #     )
    #     print("average train loss:", train_loss)
    #     print("average eval loss:", eval_loss)
    # print("step:0,lr:", get_lr(0, 100, 1000, 1e-3, 1e-4))
    # print("step:50,lr:", get_lr(50, 100, 1000, 1e-3, 1e-4))
    # print("step:100,lr:", get_lr(100, 100, 1000, 1e-3, 1e-4))
    # print("step:550,lr:", get_lr(550, 100, 1000, 1e-3, 1e-4))
    # print("step:1000,lr:", get_lr(1000, 100, 1000, 1e-3, 1e-4))
    # print("step:1100,lr:", get_lr(1100, 100, 1000, 1e-3, 1e-4))

    print(rope(8, 8))
