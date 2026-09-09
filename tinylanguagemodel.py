import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(39)


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
        self.heads = MultiHeadAttention(n_embd, num_head, block_size, dropout)
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
        dropout: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.block_size = block_size
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
        temperature: float = 0.8,
    ):

        result = idx
        if temperature <= 0:
            raise ValueError("temperature必须大于0")

        for _ in range(max_new_tokens):
            context = result[:, max(0, result.shape[1] - self.block_size) :]
            logits, _ = self(context)
            logits = logits[:, -1, :]
            k = logits.shape[-1]
            if top_k == 1:
                new_token = torch.argmax(logits, dim=-1, keepdim=True)
            elif top_k is not None and top_k <= 0:
                raise ValueError("top_k必须大于0")
            elif top_k is not None:
                k = min(top_k, logits.shape[-1])
                topk_logits, topk_indices = torch.topk(logits / temperature, k, dim=-1)
                topk_prob = torch.softmax(topk_logits, dim=-1)
                sample_position = torch.multinomial(topk_prob, 1)
                new_token = torch.gather(topk_indices, dim=-1, index=sample_position)
            else:
                prob = torch.softmax(logits / temperature, dim=-1)
                new_token = torch.multinomial(prob, 1)

            result = torch.concat((result, new_token), dim=-1)
            if new_token == eos_token:
                break
        return result


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
    patience=3,
    min_delta=0.01,
):

    model.train()

    non_improve = 0
    best_val_loss = float("inf")
    for step in range(1, 1 + steps):
        input_x, target = get_batch(
            data=train_data,
            batch_size=batch_size,
            block_size=block_size,
            device=device,
        )
        optimizer.zero_grad()
        _, loss = model(input_x, target)
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print(step, "train loss:", loss.item())
        if step % 1000 == 0:
            train_loss, eval_loss = estimate_loss(
                model=model,
                train_data=train_data,
                val_data=val_data,
                batch_size=batch_size,
                block_size=block_size,
                eval_iters=eval_iters,
                device=device,
            )
            print(f"{step}\ttrain loss:{train_loss}\teval loss:{eval_loss}")
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
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": current_train_loss,
            "val_loss": current_eval_loss,
        },
        "last_checkpoint.pt",
    )
    if current_eval_loss < best_val_loss - min_delta:
        best_val_loss = current_eval_loss
        non_improve = 0
        torch.save(
            {
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": current_train_loss,
                "val_loss": best_val_loss,
            },
            "best_checkpoint.pt",
        )
    else:
        non_improve += 1
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


# if __name__ == "__main__":
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
