"""Reads results/*.json -> plots/*.png and results_table.md"""
import os, json, glob, sys
sys.path.insert(0, '/tmp/pylibs')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
H = os.path.dirname(os.path.abspath(__file__)); os.makedirs(f'{H}/plots', exist_ok=True)
R = {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob(f'{H}/results/*.json') if not f.endswith('speed.json')}
def fv(n): return R[n]['final_val'] if n in R else float('nan')

def curves(names, title, fn, ylim=(1.2, 3.0), key='val'):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for n in names:
        if n in R: ax.plot(R[n]['steps'], R[n][key], label=f"{n} ({R[n]['final_val']:.3f})")
    ax.set_ylim(*ylim); ax.set_xlabel('iteration'); ax.set_ylabel(f'{key} loss'); ax.set_title(title); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f'{H}/plots/{fn}.png', dpi=110); plt.close(fig)

curves(['base', 'no_pos', 'no_resid', 'no_ln', 'no_attn', 'no_mlp', 'act_none'], 'Removing one component (val loss)', 'ablate_components', (1.3, 3.2))
curves(['base', 'act_relu', 'no_tie', 'no_scaled_init', 'no_attn_scale', 'no_bias'], 'Smaller design choices (val loss)', 'ablate_details', (1.35, 1.9))
curves(['layers_1', 'layers_2', 'layers_4', 'base', 'layers_8', 'layers_12'], 'Depth (val loss)', 'depth', (1.3, 2.4))
curves(['heads_1', 'heads_2', 'heads_3', 'base', 'heads_12'], 'Number of heads, n_embd fixed (val loss)', 'heads', (1.35, 1.8))
curves(['ctx_8', 'ctx_32', 'ctx_64', 'ctx_128', 'base'], 'Context length (val loss)', 'context', (1.3, 2.6))
curves(['width_192', 'base', 'width_768'], 'Width (val loss)', 'width', (1.2, 2.0))
curves(['lr_1e-4', 'lr_3e-4', 'base', 'lr_3e-3', 'lr_1e-2', 'lr_const'], 'Learning rate (val loss)', 'lr', (1.3, 2.6))
curves(['no_warmup_lr3e-3', 'warmup_lr3e-3'], 'Warmup at lr=3e-3 (val loss)', 'warmup', (1.3, 3.2))

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for n, c in (('long_drop_0', 'C3'), ('long_drop_0.2', 'C0')):
    if n in R:
        ax[0].plot(R[n]['steps'], R[n]['train'], c + '--', label=f'{n} train'); ax[0].plot(R[n]['steps'], R[n]['val'], c, label=f'{n} val')
ax[0].set_ylim(0.6, 1.8); ax[0].set_title('Dropout vs overfitting (8000 iters)'); ax[0].set_xlabel('iteration'); ax[0].legend(); ax[0].grid(alpha=.3)
for n in ('base', 'drop_0', 'drop_0.4'):
    if n in R: ax[1].plot(R[n]['steps'], R[n]['val'], label=f"{n} ({R[n]['final_val']:.3f})")
ax[1].set_ylim(1.3, 1.9); ax[1].set_title('Dropout, 2000 iters (val)'); ax[1].set_xlabel('iteration'); ax[1].legend(); ax[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig(f'{H}/plots/dropout.png', dpi=110); plt.close(fig)

# summary bar chart
order = ['base', 'no_pos', 'no_resid', 'no_ln', 'no_attn', 'no_mlp', 'act_none', 'act_relu', 'no_tie', 'no_scaled_init', 'no_attn_scale', 'no_bias', 'no_causal']
order = [o for o in order if o in R]
fig, ax = plt.subplots(figsize=(8, 4.5)); vals = [fv(o) for o in order]
ax.barh(order[::-1], vals[::-1], color=['C0' if o == 'base' else 'C1' for o in order[::-1]])
for i, v in enumerate(vals[::-1]): ax.text(v, i, f' {v:.3f}', va='center', fontsize=8)
ax.set_xlabel('final val loss (lower = better)'); ax.set_title('Ablation summary'); ax.set_xlim(1.0, max(vals) * 1.12); fig.tight_layout(); fig.savefig(f'{H}/plots/summary.png', dpi=110); plt.close(fig)

# table
base = R['base']['final_val']
lines = ['| run | params (M) | final train | final val | Δval vs base | time (s) |', '|---|---|---|---|---|---|']
for n in sorted(R, key=lambda k: (k != 'base', R[k]['final_val'])):
    r = R[n]; lines.append(f"| {n} | {r['params']/1e6:.2f} | {r['final_train']:.3f} | {r['final_val']:.3f} | {r['final_val']-base:+.3f} | {r['time_s']:.0f} |")
open(f'{H}/results_table.md', 'w').write('\n'.join(lines) + '\n'); print('\n'.join(lines))
