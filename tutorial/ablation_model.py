"""
A flag-controlled copy of nanoGPT's model.py, used ONLY for the tutorial experiments.
With all flags at their defaults it is functionally identical to model.GPT (see check_equivalence.py).
Each flag switches off / swaps exactly ONE technique so we can measure what it buys us.
"""
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class Cfg:
    block_size: int = 256
    vocab_size: int = 84
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.2
    bias: bool = True
    # ---- ablation switches (defaults == nanoGPT) ----
    pos_emb: bool = True        # learned position embeddings (wpe)
    residual: bool = True       # x = x + f(x)   (skip connections)
    layernorm: bool = True      # pre-LayerNorm (ln_1, ln_2, ln_f)
    causal: bool = True         # causal mask in attention
    attn: bool = True           # include attention sublayer
    mlp: bool = True            # include MLP sublayer
    activation: str = 'gelu'    # gelu | relu | none
    tie_weights: bool = True    # wte.weight == lm_head.weight
    scaled_init: bool = True    # std 0.02/sqrt(2*n_layer) on residual projections
    attn_scale: bool = True     # divide q.k by sqrt(head_size)


class LN(nn.Module):
    def __init__(self, n, bias, on):
        super().__init__()
        self.on = on
        if on:
            self.weight = nn.Parameter(torch.ones(n))
            self.bias = nn.Parameter(torch.zeros(n)) if bias else None
    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5) if self.on else x


class Attn(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.c_attn = nn.Linear(c.n_embd, 3 * c.n_embd, bias=c.bias)
        self.c_proj = nn.Linear(c.n_embd, c.n_embd, bias=c.bias)
        self.resid_dropout = nn.Dropout(c.dropout)
    def forward(self, x):
        B, T, C = x.size(); c = self.c; hs = C // c.n_head
        q, k, v = self.c_attn(x).split(C, dim=2)
        q, k, v = [t.view(B, T, c.n_head, hs).transpose(1, 2) for t in (q, k, v)]
        if c.attn_scale:
            # fused kernel (flash attention); scale = 1/sqrt(hs) by default
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=c.dropout if self.training else 0, is_causal=c.causal)
        else:
            att = q @ k.transpose(-2, -1)
            if c.causal:
                mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
                att = att.masked_fill(~mask, float('-inf'))
            att = F.dropout(F.softmax(att, dim=-1), c.dropout, self.training)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c_fc = nn.Linear(c.n_embd, 4 * c.n_embd, bias=c.bias)
        self.act = {'gelu': nn.GELU(), 'relu': nn.ReLU(), 'none': nn.Identity()}[c.activation]
        self.c_proj = nn.Linear(4 * c.n_embd, c.n_embd, bias=c.bias)
        self.dropout = nn.Dropout(c.dropout)
    def forward(self, x):
        return self.dropout(self.c_proj(self.act(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.ln_1 = LN(c.n_embd, c.bias, c.layernorm)
        self.attn = Attn(c) if c.attn else None
        self.ln_2 = LN(c.n_embd, c.bias, c.layernorm)
        self.mlp = MLP(c) if c.mlp else None
    def _add(self, x, f):
        return x + f if self.c.residual else f
    def forward(self, x):
        if self.attn is not None: x = self._add(x, self.attn(self.ln_1(x)))
        if self.mlp is not None:  x = self._add(x, self.mlp(self.ln_2(x)))
        return x


class GPT(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(c.vocab_size, c.n_embd),
            wpe=nn.Embedding(c.block_size, c.n_embd),
            drop=nn.Dropout(c.dropout),
            h=nn.ModuleList([Block(c) for _ in range(c.n_layer)]),
            ln_f=LN(c.n_embd, c.bias, c.layernorm),
        ))
        self.lm_head = nn.Linear(c.n_embd, c.vocab_size, bias=False)
        if c.tie_weights:
            self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init)
        std = 0.02 / math.sqrt(2 * c.n_layer) if c.scaled_init else 0.02
        for n, p in self.named_parameters():
            if n.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=std)
        if not c.pos_emb:  # keep the parameter (for identical param count in other runs) but unused
            self.transformer.wpe.weight.requires_grad_(False)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def get_num_params(self):
        # trainable params, minus position embeddings (same convention as model.py)
        n = sum(p.numel() for p in self.parameters())
        return n - self.transformer.wpe.weight.numel()

    def forward(self, idx, targets=None):
        b, t = idx.size()
        x = self.transformer.wte(idx)
        if self.config.pos_emb:
            x = x + self.transformer.wpe(torch.arange(t, device=idx.device))
        x = self.transformer.drop(x)
        for blk in self.transformer.h:
            x = blk(x)
        x = self.transformer.ln_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            return logits, F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return self.lm_head(x[:, [-1], :]), None

    def configure_optimizers(self, weight_decay, lr, betas):
        ps = {n: p for n, p in self.named_parameters() if p.requires_grad}
        groups = [{'params': [p for p in ps.values() if p.dim() >= 2], 'weight_decay': weight_decay},
                  {'params': [p for p in ps.values() if p.dim() < 2], 'weight_decay': 0.0}]
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=True)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            cond = idx[:, -self.config.block_size:]
            logits, _ = self(cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('inf')
            idx = torch.cat((idx, torch.multinomial(F.softmax(logits, -1), 1)), dim=1)
        return idx
