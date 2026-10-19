"""Speed experiments on the A100: flash vs manual attention, bf16 vs fp32, torch.compile. Writes results/speed.json"""
import sys, os, time, json, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torch.nn.functional as F
from ablation_model import GPT, Cfg
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True

def bench(label, cfg, dtype, compile_=False, B=64, steps=30):
    torch.manual_seed(0)
    m = GPT(cfg).cuda(); opt = m.configure_optimizers(0.1, 1e-3, (0.9, 0.99))
    if compile_: m = torch.compile(m)
    x = torch.randint(0, cfg.vocab_size, (B, cfg.block_size)).cuda(); y = x.clone()
    def step():
        with torch.autocast('cuda', dtype=dtype, enabled=dtype != torch.float32):
            loss = m(x, y)[1]
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    for _ in range(5): step()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t = time.time()
    for _ in range(steps): step()
    torch.cuda.synchronize(); dt = (time.time() - t) / steps
    r = dict(label=label, ms_per_iter=dt * 1e3, tokens_per_s=B * cfg.block_size / dt, peak_mem_gb=torch.cuda.max_memory_allocated() / 1e9)
    print(f"{label:34s} {r['ms_per_iter']:7.1f} ms/iter  {r['tokens_per_s']/1e3:8.0f}k tok/s  {r['peak_mem_gb']:5.1f} GB", flush=True)
    return r

out = []
base = Cfg(dropout=0.0)
out.append(bench('bf16 + fused sdpa (nanoGPT)', base, torch.bfloat16))
out.append(bench('fp32 (TF32 matmul) + fused sdpa', base, torch.float32))
out.append(bench('bf16 + manual attention', Cfg(dropout=0.0, attn_scale=False), torch.bfloat16))  # manual path (materialises TxT)
out.append(bench('bf16 + torch.compile', base, torch.bfloat16, compile_=True))
for T in (256, 1024, 2048):
    c1 = Cfg(dropout=0.0, block_size=T); c2 = Cfg(dropout=0.0, block_size=T, attn_scale=False)
    B = max(4, 64 * 256 // T)   # same tokens per iter
    out.append(bench(f'T={T} fused sdpa', c1, torch.bfloat16, B=B))
    out.append(bench(f'T={T} manual attn', c2, torch.bfloat16, B=B))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'speed.json'), 'w'), indent=1)
