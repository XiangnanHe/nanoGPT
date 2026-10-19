"""
Step-by-step, runnable tour of nanoGPT/model.py with tiny concrete examples.
Run:  python walkthrough.py        (CPU is fine; output is saved in walkthrough_output.txt)
"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import torch, torch.nn.functional as F
from model import GPT, GPTConfig, CausalSelfAttention, LayerNorm, MLP, Block
torch.manual_seed(0)
torch.set_printoptions(precision=3, linewidth=140, sci_mode=False)
def H(s): print('\n' + '=' * 78 + f'\n{s}\n' + '=' * 78)

# ---------------------------------------------------------------- 1. tokens
H('STEP 1  Text -> token ids (char-level here; GPT-2 uses BPE with vocab 50257->padded 50304)')
text = "To be, or not to be"
chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}; itos = {i: c for c, i in stoi.items()}
ids = torch.tensor([stoi[c] for c in text])
print('vocab:', chars); print('ids  :', ids.tolist())
x, y = ids[:-1], ids[1:]
print('Training pairs (input -> target = next char). One sequence of length T gives T examples at once:')
for t in range(5): print(f'   after {text[:t+1]!r:12} predict {text[t+1]!r}')

# ---------------------------------------------------------------- 2. embeddings
H('STEP 2  Embeddings: token embedding (wte) + learned position embedding (wpe)')
cfg = GPTConfig(block_size=8, vocab_size=len(chars), n_layer=2, n_head=2, n_embd=8, dropout=0.0, bias=False)
m = GPT(cfg).eval()
idx = x[None, :8]                                      # (B=1, T=8)
tok = m.transformer.wte(idx); pos = m.transformer.wpe(torch.arange(idx.size(1)))
print('idx', tuple(idx.shape), '-> tok_emb', tuple(tok.shape), '+ pos_emb', tuple(pos.shape), '(broadcast over batch)')
sp = [t for t in range(8) if text[t] == ' ']        # the space char occurs at several positions
print(f'The char {text[sp[0]]!r} occurs at t={sp}. Same wte row, different wpe rows:')
print('   tok_emb equal  :', torch.equal(tok[0, sp[0]], tok[0, sp[1]]))
print('   pos_emb equal  :', torch.equal(pos[sp[0]], pos[sp[1]]), '  -> after adding, the model can tell the two spaces apart')
print('x[0,0] = tok+pos =', (tok + pos)[0, 0])

# ---------------------------------------------------------------- 3. layernorm
H('STEP 3  LayerNorm: re-centre and re-scale each token vector (keeps activations stable)')
v = torch.randn(2, 4, 8) * 5 + 3
ln = LayerNorm(8, bias=False)
o = ln(v)
print('before: mean %.2f std %.2f   (per-token)' % (v.mean(-1)[0, 0], v.std(-1, unbiased=False)[0, 0]))
print('after : mean %.2f std %.2f' % (o.mean(-1)[0, 0], o.std(-1, unbiased=False)[0, 0]))
print('nanoGPT uses PRE-norm: x = x + attn(ln(x)). The residual path itself is never normalised, so gradients flow straight through.')

# ---------------------------------------------------------------- 4. attention by hand
H('STEP 4  Causal self-attention computed BY HAND (T=4 tokens, C=8, 2 heads) and checked against model.py')
cfg4 = GPTConfig(block_size=4, vocab_size=10, n_layer=1, n_head=2, n_embd=8, dropout=0.0, bias=False)
att = CausalSelfAttention(cfg4).eval()
xin = torch.randn(1, 4, 8)
B, T, C = xin.shape; nh = 2; hs = C // nh
q, k, v = att.c_attn(xin).split(C, dim=2)             # one matmul gives Q,K,V for all heads
print('c_attn out split ->  q,k,v each', tuple(q.shape))
q, k, v = [t.view(B, T, nh, hs).transpose(1, 2) for t in (q, k, v)]
print('after view+transpose each is (B, n_head, T, head_size) =', tuple(q.shape))
scores = q @ k.transpose(-2, -1) / math.sqrt(hs)       # (B, nh, T, T)
print('\nraw scores head0 (row = query position, col = key position):\n', scores[0, 0])
mask = torch.tril(torch.ones(T, T)).bool()
masked = scores.masked_fill(~mask, float('-inf'))
print('\nafter causal mask (-inf = cannot look at the future):\n', masked[0, 0])
w = F.softmax(masked, dim=-1)
print('\nattention weights head0 (each row sums to 1; row t only mixes tokens <= t):\n', w[0, 0])
print('row sums:', w[0, 0].sum(-1))
y_hand = (w @ v).transpose(1, 2).contiguous().view(B, T, C)
y_hand = att.c_proj(y_hand)
y_model = att(xin)
print('\nmax |hand - CausalSelfAttention| =', (y_hand - y_model).abs().max().item(), '  <- identical')
print('Why /sqrt(head_size)? q.k has variance ~hs, so without scaling softmax saturates (see experiment no_attn_scale).')
qq, kk = torch.randn(10000, 64), torch.randn(10000, 64)
print(f'   std of q.k for hs=64: {(qq*kk).sum(-1).std():.2f} (unscaled)   {((qq*kk).sum(-1)/8).std():.2f} (scaled by 1/sqrt(64))')

# ---------------------------------------------------------------- 5. causality test
H('STEP 5  Proof of causality: changing a FUTURE token must not change earlier outputs')
mm = GPT(GPTConfig(block_size=8, vocab_size=10, n_layer=2, n_head=2, n_embd=8, dropout=0.0)).eval()
a_ = torch.randint(0, 10, (1, 8)); b_ = a_.clone(); b_[0, 5] = (b_[0, 5] + 1) % 10   # change token at t=5
def all_logits(model, i):                                # full (B,T,V) logits via hook-free trick: run each prefix
    return torch.cat([model(i[:, :t + 1])[0] for t in range(i.size(1))], 1)
d = (all_logits(mm, a_) - all_logits(mm, b_)).abs().sum(-1)[0]
print('|delta logits| per position after editing token t=5:', d)
print('=> positions 0..4 unchanged (0.000), positions 5..7 change. This is why one pass trains on T next-token tasks safely.')

# ---------------------------------------------------------------- 6. MLP + block
H('STEP 6  MLP (4x expand, GELU, project back) and the Block (two residual updates)')
mlp = MLP(GPTConfig(n_embd=8, bias=False, dropout=0.0))
print('c_fc', tuple(mlp.c_fc.weight.shape), ' c_proj', tuple(mlp.c_proj.weight.shape), ' (8 -> 32 -> 8)')
print('Attention = tokens TALK to each other (mix across T). MLP = each token THINKS alone (mix across C). ~2/3 of params are MLP.')
blk = Block(GPTConfig(n_embd=8, n_head=2, bias=False, dropout=0.0, block_size=8)).eval()
xx = torch.randn(1, 8, 8)
out = blk(xx)
print('Block keeps shape', tuple(out.shape), '; Block(x)-x is the "update" written into the residual stream:')
print('   |x| = %.2f   |block(x)-x| = %.2f' % (xx.norm(), (out - xx).norm()))

# ---------------------------------------------------------------- 7. full model
H('STEP 7  Whole GPT at the real "baby" size used in our experiments (6 layers, 6 heads, 384 dim, ctx 256, vocab 84)')
big = GPT(GPTConfig(block_size=256, vocab_size=84, n_layer=6, n_head=6, n_embd=384, dropout=0.0, bias=True))
tot = sum(p.numel() for p in big.parameters())
comp = {
  'wte (tied with lm_head)': big.transformer.wte.weight.numel(),
  'wpe': big.transformer.wpe.weight.numel(),
  'attention (all layers)': sum(p.numel() for n, p in big.named_parameters() if '.attn.' in n),
  'mlp (all layers)': sum(p.numel() for n, p in big.named_parameters() if '.mlp.' in n),
  'layernorms': sum(p.numel() for n, p in big.named_parameters() if 'ln_' in n),
}
for k_, v_ in comp.items(): print(f'  {k_:28s} {v_:>12,}  {100*v_/tot:5.1f}%')
print(f'  {"TOTAL":28s} {tot:>12,}')
xb = torch.randint(0, 84, (4, 256)); yb = torch.randint(0, 84, (4, 256))
_, loss = big(xb, yb)
print(f'\nloss at init = {loss.item():.3f}  vs  ln(vocab)=ln(84)={math.log(84):.3f}  (uniform guess). Small init (std 0.02) => near-uniform logits.')
print('Weight tying saves', f'{84*384:,}', 'params here, but for GPT-2 (50304x768) it saves 38.6M of 124M (31%).')

# ---------------------------------------------------------------- 8. scaled init
H('STEP 8  Scaled init of residual projections: std 0.02/sqrt(2*n_layer)')
print('Each block ADDS two things into the residual stream, so variance grows ~ 2*n_layer. Scaling the writers keeps it ~constant.')
for L in (6, 24):
    cfgL = GPTConfig(block_size=64, vocab_size=84, n_layer=L, n_head=6, n_embd=384, dropout=0.0)
    for scaled in (True, False):
        mL = GPT(cfgL).eval()
        if not scaled:
            for n, p in mL.named_parameters():
                if n.endswith('c_proj.weight'): torch.nn.init.normal_(p, 0, 0.02)
        h = mL.transformer.wte(torch.randint(0, 84, (2, 64))) + mL.transformer.wpe(torch.arange(64))
        for b_ in mL.transformer.h: h = b_(h)
        print(f'  n_layer={L:2d} scaled_init={str(scaled):5s}  residual-stream std after last block = {h.std().item():.4f}')

# ---------------------------------------------------------------- 9. optimizer groups
H('STEP 9  configure_optimizers: weight decay only on matrices (>=2D); not on biases / LayerNorm gains')
_ = big.configure_optimizers(0.1, 1e-3, (0.9, 0.99), 'cpu')

# ---------------------------------------------------------------- 10. generate
H('STEP 10  Sampling: temperature and top-k acting on one logit vector')
logits = torch.tensor([3.0, 2.0, 1.0, 0.0, -1.0])
for T_ in (0.5, 1.0, 2.0):
    print(f'  temperature {T_}: probs =', F.softmax(logits / T_, -1))
v_, _ = torch.topk(logits, 2); l2 = logits.clone(); l2[l2 < v_[-1]] = -float('inf')
print('  top_k=2 (T=1): probs =', F.softmax(l2, -1), '  <- tail tokens can never be sampled')
print('generate() is autoregressive: crop to block_size, forward, take LAST position logits, sample, append, repeat.')
