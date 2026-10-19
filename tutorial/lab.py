"""
Shared helpers for the notebook and the dashboard (so both show the same numbers).
  load_checkpoint / attention_maps / generate_text   -> poke at a trained model
  param_breakdown                                    -> where do the parameters / FLOPs go?
  load_results / results_frame                       -> the 40 saved experiments
  run_live                                           -> launch a fresh training run and stream its loss
"""
import os, sys, re, json, glob, math, subprocess
import torch, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..'))
from ablation_model import GPT, Cfg

PYTHON = sys.executable
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


# ------------------------------------------------------------------ models
def load_checkpoint(name='ckpt_base', device=DEV):
    path = os.path.join(HERE, 'checkpoints', f'{name}.pt')
    if not os.path.exists(path):
        return None, None
    ck = torch.load(path, map_location=device, weights_only=True)
    model = GPT(Cfg(**ck['cfg'])).to(device)
    model.load_state_dict(ck['model']); model.eval()
    return model, {'itos': ck['itos'], 'stoi': ck['stoi']}


def encode(meta, text):
    return [meta['stoi'][c] for c in text if c in meta['stoi']]

def decode(meta, ids):
    return ''.join(meta['itos'][i] for i in ids)


@torch.no_grad()
def attention_maps(model, idx):
    """Returns a list (one entry per layer) of attention weights, each (n_head, T, T), for the first sequence in idx."""
    c = model.config
    B, T = idx.shape; C = c.n_embd; hs = C // c.n_head
    x = model.transformer.wte(idx) + model.transformer.wpe(torch.arange(T, device=idx.device))
    maps = []
    mask = torch.tril(torch.ones(T, T, device=idx.device, dtype=torch.bool))
    for blk in model.transformer.h:
        q, k, v = blk.attn.c_attn(blk.ln_1(x)).split(C, dim=2)
        q = q.view(B, T, c.n_head, hs).transpose(1, 2); k = k.view(B, T, c.n_head, hs).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hs)
        att = att.masked_fill(~mask, float('-inf')).softmax(-1)
        maps.append(att[0].float().cpu())
        x = blk(x)
    return maps


@torch.no_grad()
def next_char_probs(model, meta, text, top=8):
    idx = torch.tensor([encode(meta, text)[-model.config.block_size:]], device=next(model.parameters()).device)
    logits, _ = model(idx)
    p = F.softmax(logits[0, -1].float(), -1)
    v, i = torch.topk(p, top)
    return [(repr(meta['itos'][j.item()]), round(x.item(), 4)) for x, j in zip(v, i)]


@torch.no_grad()
def generate_text(model, meta, prompt='\n', n=300, temperature=1.0, top_k=40, seed=0):
    torch.manual_seed(seed)
    ids = encode(meta, prompt) or [meta['stoi']['\n']]
    idx = torch.tensor([ids], device=next(model.parameters()).device)
    out = model.generate(idx, n, temperature=max(temperature, 1e-4), top_k=top_k if top_k and top_k > 0 else None)
    return decode(meta, out[0].tolist())


# ------------------------------------------------------------------ calculator
def param_breakdown(n_layer=6, n_head=6, n_embd=384, block_size=256, vocab_size=84, bias=True, tie=True):
    C, V, T, L = n_embd, vocab_size, block_size, n_layer
    b = 1 if bias else 0
    attn = L * (C * 3 * C + b * 3 * C + C * C + b * C)
    mlp = L * (C * 4 * C + b * 4 * C + 4 * C * C + b * C)
    ln = (2 * L + 1) * C * (1 + b)
    wte, wpe = V * C, T * C
    head = 0 if tie else V * C
    total = wte + wpe + attn + mlp + ln + head
    flops_per_token = 6 * (total - wpe) + 12 * L * n_head * (C // n_head) * T   # same formula as GPT.estimate_mfu
    return dict(token_emb=wte, pos_emb=wpe, attention=attn, mlp=mlp, layernorm=ln, lm_head_extra=head, total=total,
                flops_per_token=flops_per_token, train_state_mb=total * 16 / 1e6)  # fp32 weights+grads+2 Adam moments


# ------------------------------------------------------------------ results
def load_results():
    out = {}
    for f in glob.glob(os.path.join(HERE, 'results', '*.json')):
        n = os.path.basename(f)[:-5]
        if n != 'speed':
            out[n] = json.load(open(f))
    return out

def results_frame():
    import pandas as pd
    R = load_results(); base = R['base']['final_val']
    rows = [dict(run=n, params_M=round(r['params'] / 1e6, 2), train=round(r['final_train'], 3), val=round(r['final_val'], 3),
                 delta_vs_base=round(r['final_val'] - base, 3), best_val=round(r['best_val'], 3), iters=r['hp']['iters'], seconds=round(r['time_s']))
            for n, r in R.items()]
    return pd.DataFrame(rows).sort_values('val').reset_index(drop=True)


# ------------------------------------------------------------------ live training
LINE = re.compile(r'it\s+(\d+) train ([\d.]+) val ([\d.]+)')

def run_live(name, **kw):
    """Launch run_experiment.py and yield (iter, train, val) as it logs. kw: any Cfg / hyper-param, e.g. residual=False, iters=400"""
    args = [PYTHON, os.path.join(HERE, 'run_experiment.py'), name] + [f'{k}={v}' for k, v in kw.items()]
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=HERE)
    tail = []
    for line in p.stdout:
        tail.append(line)
        m = LINE.search(line)
        if m:
            yield int(m.group(1)), float(m.group(2)), float(m.group(3))
    p.wait()
    if p.returncode != 0:
        raise RuntimeError('run failed:\n' + ''.join(tail[-15:]))

def quick_run(name, iters=300, **kw):
    """Blocking helper for notebooks: returns dict(steps, train, val)."""
    kw = dict(iters=iters, eval_every=max(iters // 10, 10), eval_iters=20, **kw)
    s, tr, va = [], [], []
    for it, t, v in run_live('live_' + name, **kw):
        s.append(it); tr.append(t); va.append(v)
    return dict(steps=s, train=tr, val=va)


# ------------------------------------------------------------------ first-time setup
DATA_DIR = os.path.join(HERE, '..', 'data', 'shakespeare_plays')

def ensure_data():
    """Build train.bin/val.bin if missing: fetch the PyPI sdist `shakespeare==0.6` (only PyPI is reachable from Bolt), unpack into a fresh temp dir."""
    if os.path.exists(os.path.join(DATA_DIR, 'train.bin')):
        return 'data ready'
    import tempfile, tarfile
    tmp = tempfile.mkdtemp(prefix='shk_')
    dl, ex = os.path.join(tmp, 'dl'), os.path.join(tmp, 'ex')
    os.makedirs(dl); os.makedirs(ex)
    subprocess.check_call([PYTHON, '-m', 'pip', 'download', 'shakespeare==0.6', '--no-deps', '-q', '-d', dl])
    tar = glob.glob(os.path.join(dl, '*.tar.gz'))[0]
    with tarfile.open(tar) as t:
        t.extractall(ex, filter='data')
    texts = glob.glob(os.path.join(ex, '*', 'shksprdata', 'texts'))[0]
    subprocess.check_call([PYTHON, '-I', os.path.join(DATA_DIR, 'prepare.py'), texts])
    return 'data built'

def ensure_checkpoint(name='ckpt_base', iters=3000):
    """Train the reference baby-GPT (about 95 s on an A100) if no checkpoint exists."""
    if load_checkpoint(name)[0] is None:
        ensure_data()
        for it, t, v in run_live(name, iters=iters, save=True):
            print(f'  it {it:5d}  train {t:.3f}  val {v:.3f}')
    return load_checkpoint(name)
