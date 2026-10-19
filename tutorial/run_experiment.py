"""
Train one baby-GPT configuration and log train/val loss to results/<name>.json.
usage: python run_experiment.py NAME [key=value ...]     (keys: any Cfg field or training hyper-param)
Training recipe mirrors config/train_shakespeare_char.py (AdamW, warmup + cosine decay, bf16 autocast, grad clip 1.0).
"""
import sys, os, json, time, math, pickle
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ablation_model import GPT, Cfg

HERE = os.path.dirname(os.path.abspath(__file__))
name = sys.argv[1]
hp = dict(iters=2000, batch_size=64, lr=1e-3, min_lr=1e-4, warmup=100, wd=0.1, beta2=0.99, eval_every=100, eval_iters=50,
          seed=1337, schedule='cosine', dataset='shakespeare_plays', sample=False, save=False)
cfg_kw = {}
for a in sys.argv[2:]:
    k, v = a.split('=')
    tgt = hp if k in hp else cfg_kw
    assert k in hp or k in Cfg.__dataclass_fields__, k
    ref = hp[k] if k in hp else getattr(Cfg(), k)
    tgt[k] = (v.lower() == 'true') if isinstance(ref, bool) else type(ref)(v)

torch.manual_seed(hp['seed']); np.random.seed(hp['seed'])
dev = 'cuda'
ddir = os.path.join(HERE, '..', 'data', hp['dataset'])
meta = pickle.load(open(os.path.join(ddir, 'meta.pkl'), 'rb'))
cfg = Cfg(vocab_size=meta['vocab_size'], **cfg_kw)
train = np.memmap(os.path.join(ddir, 'train.bin'), dtype=np.uint16, mode='r')
val = np.memmap(os.path.join(ddir, 'val.bin'), dtype=np.uint16, mode='r')

def batch(split):
    d = train if split == 'train' else val
    ix = torch.randint(len(d) - cfg.block_size - 1, (hp['batch_size'],))
    x = torch.stack([torch.from_numpy(d[i:i + cfg.block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(d[i + 1:i + 1 + cfg.block_size].astype(np.int64)) for i in ix])
    return x.to(dev), y.to(dev)

model = GPT(cfg).to(dev)
opt = model.configure_optimizers(hp['wd'], hp['lr'], (0.9, hp['beta2']))
def get_lr(it):
    if hp['schedule'] == 'constant': return hp['lr']
    if it < hp['warmup']: return hp['lr'] * (it + 1) / (hp['warmup'] + 1)
    r = (it - hp['warmup']) / (hp['iters'] - hp['warmup'])
    return hp['min_lr'] + 0.5 * (1 + math.cos(math.pi * r)) * (hp['lr'] - hp['min_lr'])

@torch.no_grad()
def evaluate():
    model.eval(); out = {}
    for s in ('train', 'val'):
        ls = []
        for _ in range(hp['eval_iters']):
            x, y = batch(s)
            with torch.autocast('cuda', dtype=torch.bfloat16): ls.append(model(x, y)[1].item())
        out[s] = float(np.mean(ls))
    model.train(); return out

log = dict(name=name, hp=hp, cfg=cfg.__dict__, params=model.get_num_params(), steps=[], train=[], val=[])
t0 = time.time(); model.train()
for it in range(hp['iters'] + 1):
    if it % hp['eval_every'] == 0 or it == hp['iters']:
        e = evaluate(); log['steps'].append(it); log['train'].append(e['train']); log['val'].append(e['val'])
        print(f"[{name}] it {it:5d} train {e['train']:.4f} val {e['val']:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    if it == hp['iters']: break
    for g in opt.param_groups: g['lr'] = get_lr(it)
    x, y = batch('train')
    with torch.autocast('cuda', dtype=torch.bfloat16): loss = model(x, y)[1]
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); opt.zero_grad(set_to_none=True)
    if not math.isfinite(loss.item()): print('diverged'); log['diverged'] = True; break
log['time_s'] = time.time() - t0
log['final_val'] = log['val'][-1]; log['best_val'] = min(log['val']); log['final_train'] = log['train'][-1]
if hp['sample']:
    model.eval(); itos = meta['itos']
    out = model.generate(torch.tensor([[meta['stoi']['\n']]], device=dev), 400, top_k=40)[0].tolist()
    log['sample'] = ''.join(itos[i] for i in out)
if hp['save']:
    os.makedirs(os.path.join(HERE, 'checkpoints'), exist_ok=True)
    torch.save({'model': model.state_dict(), 'cfg': cfg.__dict__, 'itos': meta['itos'], 'stoi': meta['stoi']}, os.path.join(HERE, 'checkpoints', f'{name}.pt'))
os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
json.dump(log, open(os.path.join(HERE, 'results', f'{name}.json'), 'w'))
print(f"[{name}] DONE final val {log['final_val']:.4f} best {log['best_val']:.4f} time {log['time_s']:.0f}s")
