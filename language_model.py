import torch
import torch.nn as nn
from torch.nn import functional as F

from params import (
    batch_size, block_size, device, eval_iters,
    dropout, n_layers, n_embds, n_head
)

def get_batch(data):
    """ Generate a batch of inputs (x) and targets (y) """
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def loss_estimation(model, train_data, test_data):
    output = {}
    model.eval()
    for learn_env in ['train', 'test']:
        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):
            X, Y = get_batch(train_data if learn_env == 'train' else test_data)
            _, loss = model(X, Y)
            losses[k] = loss.item()
            output[learn_env] = losses.mean()
    model.train()
    return output


class Head(nn.Module):
    """ One head of self attention """

    def __init__(self, head_size: int) -> None:
        super().__init__()
        self.key = nn.Linear(n_embds, head_size, bias=False)
        self.query = nn.Linear(n_embds, head_size, bias=False)
        self.value = nn.Linear(n_embds, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, _x):
        _, TIME, CHANNEL = _x.shape
        key = self.key(_x)
        query = self.query(_x)

        # calculate affinities
        wei = query @ key.transpose(-2, -1) * CHANNEL**-0.5  # (B,T,C) @ (B,C,T) => (B,T,T)
        wei = wei.masked_fill(self.tril[:TIME, :TIME] == 0, float('-inf'))  # type: ignore # B,T,T
        wei = F.softmax(wei, dim=-1)  # B,T,T
        wei = self.dropout(wei)

        # weighted aggregation
        v = self.value(_x)  # B,T,C
        output = wei @ v  # (B,T,T) @ (B,T,C) => (B,T,C)

        return output


class MultiHead(nn.Module):
    """ Multi head of self attention for rinning in parallel """

    def __init__(self, n_heads: int, head_size: int) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embds, n_embds)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        output = torch.cat([head(x) for head in self.heads], dim=-1)
        output = self.dropout(self.proj(output))
        return output


class FeedForward(nn.Module):
    """ Linear layer followed by non linearity """

    def __init__(self, n_embds: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embds, 4 * n_embds),
            nn.ReLU(),
            nn.Linear(4 * n_embds, n_embds),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block, for communication followed by computation """

    def __init__(self, n_embds: int, n_head: int) -> None:
        super().__init__()
        head_size = n_embds // n_head
        self.sa = MultiHead(n_head, head_size)
        self.ffwd = FeedForward(n_embds)
        self.ln1 = nn.LayerNorm(n_embds)
        self.ln2 = nn.LayerNorm(n_embds)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))

        return x


class BigramLangModel(nn.Module):

    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_embd_table = nn.Embedding(vocab_size, n_embds)
        self.pos_embd_table = nn.Embedding(block_size, n_embds)
        self.blocks = nn.Sequential(*[Block(n_embds, n_head=n_head) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embds)
        self.l_head = nn.Linear(n_embds, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None =None):
        B, T = idx.shape

        token_embd = self.token_embd_table(idx)  # B,T,C
        pos_embd = self.pos_embd_table(torch.arange(T, device=device))  # T,C

        x = token_embd + pos_embd  # B,T,C
        x = self.blocks(x)  # B,T,C
        x = self.ln_f(x)  # B,T,C

        logits = self.l_head(x)  # B,T,vocab_size

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx: list, max_new_tokens: int):
        for _ in range(max_new_tokens):
            idx_crop = idx[:, -block_size:]
            logits, loss = self(idx_crop)
            logits = logits[:, -1,  :]  # B,C
            probabilities = F.softmax(logits, dim=-1)  # B,C
            idx_next = torch.multinomial(probabilities, num_samples=1)  # B, 1
            idx = torch.cat((idx, idx_next), dim=1)  # B, T+1

        return idx

