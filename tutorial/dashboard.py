"""
nanoGPT tutorial dashboard.   Run:  streamlit run dashboard.py --server.port <PORT> --server.address 0.0.0.0
Tabs: model.py map | experiments | attention lab | generate | live training lab | size calculator | speed
"""
import os, sys, json, time
import streamlit as st
import numpy as np, pandas as pd, torch
import plotly.express as px, plotly.graph_objects as go

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lab
from ablation_model import Cfg

st.set_page_config(page_title='nanoGPT lab', page_icon='🧠', layout='wide')
st.title('🧠 nanoGPT lab — understand `model.py` by poking at it')

@st.cache_resource(show_spinner='loading reference model…')
def get_model():
    return lab.load_checkpoint('ckpt_base')

@st.cache_data
def get_results():
    return lab.load_results()

R = get_results()
model, meta = get_model()
tabs = st.tabs(['📖 model.py map', '📊 Experiments', '🔍 Attention lab', '✍️ Generate', '🧪 Live training lab', '🧮 Size calculator', '⚡ Speed'])

# ---------------------------------------------------------------- 0. model.py map
with tabs[0]:
    st.subheader('Read `model.py` piece by piece')
    src = open(os.path.join(HERE, '..', 'model.py')).read().splitlines()
    PARTS = [
        ('LayerNorm', 18, 27, 'Normalises each token vector (mean 0, std 1) then applies a learned gain/bias. Keeps activations at a stable scale so deep stacks train. Ablation: removing it costs **+0.15** val loss.'),
        ('CausalSelfAttention', 29, 76, 'One Linear makes Q,K,V for all heads. Scores `q·kᵀ/√hs`, **causal mask** (future = -inf), softmax, weighted sum of V, merge heads, output projection. Removing attention: **+1.15**. Removing the mask: val loss → 0.01 (it just copies the answer = leakage). Removing `1/√hs`: **+0.10**.'),
        ('MLP', 78, 92, 'Expand 4×, GELU, project back; processes every position independently. ~2/3 of all parameters. Removing it: **+0.39**; removing the non-linearity: **+0.34**.'),
        ('Block', 94, 106, 'Two **residual** updates with **pre-norm**: `x += attn(ln(x)); x += mlp(ln(x))`. Removing residuals: the model cannot learn beyond character frequencies (val 3.34 vs 1.40).'),
        ('GPTConfig', 108, 116, 'The hyper-parameters. Our baby model: 6 layers, 6 heads, 384 dims, context 256, vocab 84.'),
        ('GPT.__init__ (weights, tying, init)', 118, 148, '`wte`/`wpe` embeddings, blocks, final LN, `lm_head`. **Weight tying** shares wte and lm_head (31% of GPT-2 124M). **Scaled init** of residual projections `0.02/√(2·n_layer)` keeps the residual-stream variance flat with depth (no measurable effect at 6 layers).'),
        ('GPT.forward', 170, 193, 'tok_emb + pos_emb → blocks → ln_f → lm_head → cross-entropy. At inference only the **last** position goes through lm_head. Removing pos-emb: **+0.26**.'),
        ('configure_optimizers', 263, 287, 'AdamW; weight-decay only on ≥2-D tensors (matrices, embeddings), none on biases/LayerNorm gains. Fused kernel on CUDA.'),
        ('estimate_mfu', 289, 303, 'FLOPs/token = `6N + 12·L·H·Q·T`, relative to A100 bf16 peak 312 TFLOPS. Our baby model reaches ~1M tok/s.'),
        ('generate', 305, 330, 'Autoregressive loop: crop to block_size → forward → last logits → /temperature → top-k → softmax → sample → append.'),
    ]
    name = st.radio('Component', [p[0] for p in PARTS], horizontal=True)
    n, a, b, note = next(p for p in PARTS if p[0] == name)
    c1, c2 = st.columns([3, 2])
    c1.code('\n'.join(src[a - 1:b]), language='python', line_numbers=True)
    c2.markdown(f'**Lines {a}–{b}**\n\n{note}')
    c2.info('Next: try the *Attention lab* and *Live training lab* tabs to see these claims for yourself, or open `nanogpt_tutorial.ipynb` to re-implement each class.')

# ---------------------------------------------------------------- 1. experiments
with tabs[1]:
    st.subheader('40 saved experiments (A100, 2000 iterations, baby GPT on Shakespeare)')
    GROUPS = {
        'Essential components': ['base', 'no_resid', 'no_attn', 'no_mlp', 'act_none', 'no_pos', 'no_ln'],
        'Smaller design choices': ['base', 'act_relu', 'no_tie', 'no_scaled_init', 'no_attn_scale', 'no_bias', 'wd_0'],
        'Depth': ['layers_1', 'layers_2', 'layers_4', 'base', 'layers_8', 'layers_12'],
        'Heads': ['heads_1', 'heads_2', 'heads_3', 'base', 'heads_12'],
        'Width': ['width_192', 'base', 'width_768'],
        'Context length': ['ctx_8', 'ctx_32', 'ctx_64', 'ctx_128', 'base'],
        'Dropout (short)': ['drop_0', 'base', 'drop_0.4'],
        'Dropout (8000 it, overfitting)': ['long_drop_0', 'long_drop_0.2'],
        'Learning rate': ['lr_1e-4', 'lr_3e-4', 'base', 'lr_3e-3', 'lr_1e-2', 'lr_const'],
        'Warm-up @ lr 3e-3': ['no_warmup_lr3e-3', 'warmup_lr3e-3'],
        'Seed noise': ['base', 'base_s1', 'base_s2'],
        'Causal mask (leakage!)': ['base', 'no_causal'],
    }
    g = st.selectbox('Preset', list(GROUPS))
    runs = st.multiselect('Runs', sorted(R), default=[r for r in GROUPS[g] if r in R])
    c1, c2, c3 = st.columns(3)
    metric = c1.radio('Curve', ['val', 'train', 'both'], horizontal=True)
    ymax = c2.slider('y-axis max', 1.5, 5.0, 3.0, .1)
    smooth = c3.checkbox('log y-axis', False)
    fig = go.Figure()
    for r in runs:
        for m in (['val', 'train'] if metric == 'both' else [metric]):
            fig.add_trace(go.Scatter(x=R[r]['steps'], y=R[r][m], name=f'{r} ({m})', mode='lines', line=dict(dash='dot' if m == 'train' else 'solid')))
    fig.update_layout(height=450, xaxis_title='iteration', yaxis_title='loss', yaxis=dict(range=[np.log10(1.0) if smooth else 1.0, np.log10(ymax) if smooth else ymax], type='log' if smooth else 'linear'), margin=dict(t=20))
    st.plotly_chart(fig, width='stretch')
    df = lab.results_frame()
    sel = df[df.run.isin(runs)]
    st.dataframe(sel, width='stretch', hide_index=True)
    st.caption('Seed noise ≈ ±0.003 (3 baseline seeds: 1.397 / 1.399 / 1.395). Everything else is one seed, so treat |Δ| < 0.01 as “no measurable effect”. `no_causal` ≈ 0.01 is data leakage, not a good model.')
    with st.expander('All runs'):
        st.dataframe(df, width='stretch', hide_index=True)

# ---------------------------------------------------------------- 2. attention lab
with tabs[2]:
    st.subheader('What does each head look at?')
    if model is None:
        st.warning('No checkpoint found. Train one: `python run_experiment.py ckpt_base iters=3000 save=true` (≈95 s).')
    else:
        cfg = model.config
        text = st.text_area('Text', 'ROMEO.\nWhat light through yonder window breaks?', height=80)
        ids = torch.tensor([lab.encode(meta, text)[-cfg.block_size:]], device=lab.DEV)
        if ids.numel() < 2:
            st.stop()
        maps = lab.attention_maps(model, ids)
        chars = [lab.decode(meta, [i]).replace('\n', '⏎').replace(' ', '␣') for i in ids[0].tolist()]
        mode = st.radio('View', ['one head', 'all heads in a layer', 'average over heads for every layer'], horizontal=True)
        L = st.slider('Layer', 0, cfg.n_layer - 1, 0)
        def heat(A, title, h=520):
            f = px.imshow(A.numpy(), x=chars, y=chars, color_continuous_scale='magma', aspect='auto', labels=dict(x='attended-to (key)', y='current char (query)', color='weight'))
            f.update_layout(title=title, height=h, margin=dict(t=40)); return f
        if mode == 'one head':
            H = st.slider('Head', 0, cfg.n_head - 1, 0)
            st.plotly_chart(heat(maps[L][H], f'layer {L} head {H}', 620), width='stretch')
        elif mode == 'all heads in a layer':
            cols = st.columns(3)
            for h in range(cfg.n_head):
                cols[h % 3].plotly_chart(heat(maps[L][h], f'layer {L} head {h}', 330), width='stretch')
        else:
            cols = st.columns(3)
            for l in range(cfg.n_layer):
                cols[l % 3].plotly_chart(heat(maps[l].mean(0), f'layer {l} (mean of heads)', 330), width='stretch')
        st.markdown('**Things to look for:** a bright diagonal = attends to itself; a diagonal one below = previous character; vertical stripes = a “sink” token (often newline/space); each row sums to 1 and is zero above the diagonal (the causal mask).')
        st.markdown('#### What does the model predict next?')
        pr = lab.next_char_probs(model, meta, text, top=10)
        st.bar_chart(pd.DataFrame(pr, columns=['char', 'prob']).set_index('char'))

# ---------------------------------------------------------------- 3. generate
with tabs[3]:
    st.subheader('Sample from the trained model')
    if model is None:
        st.warning('No checkpoint found.')
    else:
        c1, c2 = st.columns([2, 1])
        prompt = c1.text_area('Prompt', 'ROMEO.\n', height=100)
        temp = c2.slider('Temperature', 0.1, 2.0, 0.8, .05)
        topk = c2.slider('top-k (0 = off)', 0, 84, 40)
        n = c2.slider('New characters', 50, 1500, 400, 50)
        seed = c2.number_input('Seed', 0, 9999, 0)
        st.code(lab.generate_text(model, meta, prompt, n, temp, topk, int(seed)), language=None)
        st.markdown('Low temperature → repetitive but well-formed; high temperature → creative but misspelled. top-k stops rare wrong characters from ever being picked.')
        logits = st.text_input('Playground: logits', '3, 2, 1, 0, -1')
        try:
            l = torch.tensor([float(x) for x in logits.split(',')])
            if topk and topk < len(l):
                l = torch.where(l < torch.topk(l, topk).values[-1], torch.tensor(-float('inf')), l)
            p = torch.softmax(l / temp, -1)
            st.bar_chart(pd.DataFrame({'prob': p.numpy()}))
        except Exception as e:
            st.caption(f'could not parse: {e}')

# ---------------------------------------------------------------- 4. live training lab
with tabs[4]:
    st.subheader('Train a small model right now and watch the loss')
    st.caption('Each run is a fresh process on the GPU. 300 iterations ≈ 10 s on the A100. Compare against the baseline of the same length.')
    if 'history' not in st.session_state:
        st.session_state.history = {}
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('**Techniques** (untick to remove)')
        residual = st.checkbox('residual connections', True); layernorm = st.checkbox('LayerNorm', True); pos_emb = st.checkbox('position embeddings', True)
        causal = st.checkbox('causal mask (off = leakage!)', True); attn = st.checkbox('attention', True); mlp = st.checkbox('MLP', True)
        tie = st.checkbox('weight tying', True); sinit = st.checkbox('scaled residual init', True); ascale = st.checkbox('1/√d attention scale (manual path)', True)
        act = st.selectbox('MLP activation', ['gelu', 'relu', 'none'])
    with c2:
        st.markdown('**Size**')
        n_layer = st.slider('n_layer', 1, 12, 6); n_head = st.select_slider('n_head', [1, 2, 3, 6, 12], 6); n_embd = st.select_slider('n_embd', [96, 192, 384, 768], 384)
        block = st.select_slider('block_size', [8, 32, 64, 128, 256], 256); dropout = st.slider('dropout', 0.0, 0.5, 0.2, .05)
    with c3:
        st.markdown('**Optimisation**')
        iters = st.select_slider('iterations', [100, 200, 300, 500, 1000, 2000], 300)
        lr = st.select_slider('learning rate', [1e-4, 3e-4, 1e-3, 3e-3, 1e-2], 1e-3)
        warmup = st.select_slider('warm-up iters', [0, 10, 50, 100], 100); sched = st.selectbox('schedule', ['cosine', 'constant'])
        wd = st.select_slider('weight decay', [0.0, 0.01, 0.1], 0.1); seed = st.number_input('seed', 0, 99999, 1337, key='lab_seed')
        label = st.text_input('Run label', f'run{len(st.session_state.history) + 1}')
    kw = dict(iters=iters, eval_every=max(iters // 15, 10), eval_iters=20, residual=residual, layernorm=layernorm, pos_emb=pos_emb, causal=causal, attn=attn, mlp=mlp,
              tie_weights=tie, scaled_init=sinit, attn_scale=ascale, activation=act, n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block,
              dropout=dropout, lr=lr, min_lr=lr / 10, warmup=warmup, schedule=sched, wd=wd, seed=int(seed))
    if n_embd % n_head:
        st.error('n_embd must be divisible by n_head')
    else:
        cc = st.columns([1, 1, 3])
        run_clicked = cc[0].button('▶ Run', type='primary')
        if cc[1].button('Clear history'):
            st.session_state.history = {}
        chart = st.empty(); status = st.empty()
        def draw(extra=None):
            f = go.Figure()
            for k, v in {**st.session_state.history, **({label: extra} if extra else {})}.items():
                f.add_trace(go.Scatter(x=v['steps'], y=v['val'], name=f'{k} (val)', mode='lines+markers'))
            f.update_layout(height=420, xaxis_title='iteration', yaxis_title='val loss', yaxis=dict(range=[1.0, 4.6]), margin=dict(t=20)); chart.plotly_chart(f, width='stretch')
        if run_clicked:
            cur = dict(steps=[], train=[], val=[])
            try:
                for it, t, v in lab.run_live('live_dash', **kw):
                    cur['steps'].append(it); cur['train'].append(t); cur['val'].append(v)
                    status.write(f'iter {it}/{iters}   train {t:.3f}   val {v:.3f}'); draw(cur)
                st.session_state.history[label] = cur
                status.success(f"{label}: final val {cur['val'][-1]:.3f}" + (' ⚠ val ≈ 0 means the model sees the answer (leakage).' if cur['val'][-1] < 0.3 else ''))
            except Exception as e:
                status.error(str(e))
        else:
            draw()
        st.caption('Tips: run the defaults first as your baseline, then untick one technique at a time. Try `causal` off to see the “too good to be true” loss.')

# ---------------------------------------------------------------- 5. size calculator
with tabs[5]:
    st.subheader('Where do the parameters and compute go?')
    PRE = {'our baby GPT': (6, 6, 384, 256, 84), 'GPT-2 small (124M)': (12, 12, 768, 1024, 50304), 'GPT-2 medium (350M)': (24, 16, 1024, 1024, 50304),
           'GPT-2 large (774M)': (36, 20, 1280, 1024, 50304), 'GPT-2 XL (1.5B)': (48, 25, 1600, 1024, 50304), 'custom': None}
    p = st.selectbox('Preset', list(PRE), index=0)
    base = PRE[p] or (6, 6, 384, 256, 84)
    c = st.columns(5)
    nl = c[0].number_input('n_layer', 1, 96, base[0]); nh = c[1].number_input('n_head', 1, 64, base[1]); ne = c[2].number_input('n_embd', 16, 8192, base[2], step=16)
    bs = c[3].number_input('block_size', 8, 8192, base[3]); vs = c[4].number_input('vocab_size', 8, 200000, base[4])
    c = st.columns(2); bias = c[0].checkbox('bias', True, key='calc_bias'); tie = c[1].checkbox('weight tying', True, key='calc_tie')
    if ne % nh:
        st.error('n_embd must be divisible by n_head')
    else:
        pb = lab.param_breakdown(nl, nh, ne, bs, vs, bias, tie)
        m = st.columns(4)
        m[0].metric('Parameters', f"{pb['total']/1e6:.2f} M"); m[1].metric('MLP share', f"{100*pb['mlp']/pb['total']:.0f}%")
        m[2].metric('FLOPs / token (fwd+bwd)', f"{pb['flops_per_token']/1e6:.0f} M"); m[3].metric('fp32 train state', f"{pb['train_state_mb']:.0f} MB")
        parts = pd.DataFrame({'part': ['token embedding', 'position embedding', 'attention', 'MLP', 'LayerNorm', 'extra lm_head (untied)'],
                              'params': [pb['token_emb'], pb['pos_emb'], pb['attention'], pb['mlp'], pb['layernorm'], pb['lm_head_extra']]})
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.pie(parts, names='part', values='params', hole=.4), width='stretch')
        c2.dataframe(parts.assign(share=(100 * parts.params / pb['total']).round(2)), hide_index=True, width='stretch')
        st.caption('Check: for GPT-2 small this gives ≈124.4M with tying (what the paper calls 124M). Formula for compute per token matches `GPT.estimate_mfu` (PaLM appendix B). Memory counts weights + grads + 2 Adam moments in fp32 (16 bytes/param) and ignores activations.')
        T = np.array([64, 128, 256, 512, 1024, 2048, 4096])
        qk = 12 * nl * ne * T                     # attention-matrix FLOPs per token = 12*L*H*Q*T
        base_f = 6 * (pb['total'] - pb['pos_emb'])
        st.plotly_chart(px.line(pd.DataFrame({'context T': T, 'attention FLOPs share %': 100 * qk / (qk + base_f)}), x='context T', y='attention FLOPs share %', markers=True, log_x=True,
                                title='Share of compute spent on the T×T attention term grows with context length'), width='stretch')

# ---------------------------------------------------------------- 6. speed
with tabs[6]:
    st.subheader('Speed on the A100 (training step, B×T = 16,384 tokens)')
    sp = os.path.join(HERE, 'results', 'speed.json')
    if os.path.exists(sp):
        S = pd.DataFrame(json.load(open(sp)))
        st.dataframe(S.round(1), hide_index=True, width='stretch')
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.bar(S, x='label', y='tokens_per_s', title='tokens / s'), width='stretch')
        c2.plotly_chart(px.bar(S, x='label', y='peak_mem_gb', title='peak GPU memory (GB)'), width='stretch')
        st.caption('Fused (flash) attention keeps memory flat as context grows; the manual T×T path grows quadratically (10.7 GB vs 1.7 GB at T=2048).')
    else:
        st.info('Run `python bench_speed.py` first.')
