#!/bin/bash
# Runs every experiment used in the tutorial (about 40 runs x ~1 min on one A100).
cd "$(dirname "$0")"
PY=${PY:-/miniforge/envs/iris/bin/python}
run() { [ -f results/$1.json ] && return; $PY run_experiment.py "$@" 2>&1 | grep -E "DONE|diverged|Error|error" ; }
# 0. noise floor: same config, different seeds
run base_s1 seed=1; run base_s2 seed=2
# 1. architecture ablations (each removes/changes ONE technique)
run no_pos pos_emb=false
run no_resid residual=false
run no_ln layernorm=false
run no_causal causal=false
run no_attn attn=false
run no_mlp mlp=false
run act_relu activation=relu
run act_none activation=none
run no_tie tie_weights=false
run no_scaled_init scaled_init=false
run no_attn_scale attn_scale=false
run no_bias bias=false
# 2. size: depth, heads, width
for L in 1 2 4 8 12; do run layers_$L n_layer=$L; done
for H in 1 2 3 12; do run heads_$H n_head=$H; done
run width_192 n_embd=192; run width_768 n_embd=768
# 3. context length
for T in 8 32 64 128; do run ctx_$T block_size=$T; done
# 4. regularisation: dropout (short run, then long run to expose overfitting)
run drop_0 dropout=0.0; run drop_0.4 dropout=0.4
run long_drop_0 dropout=0.0 iters=8000 eval_every=200
run long_drop_0.2 dropout=0.2 iters=8000 eval_every=200
# 5. optimisation
for LR in 1e-4 3e-4 3e-3 1e-2; do run lr_$LR lr=$LR min_lr=$(python3 -c "print($LR/10)"); done
run lr_const schedule=constant
run no_warmup_lr3e-3 warmup=0 lr=3e-3 min_lr=3e-4
run warmup_lr3e-3 lr=3e-3 min_lr=3e-4
run wd_0 wd=0.0
echo ALL_DONE
