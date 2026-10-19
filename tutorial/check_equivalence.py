"""Sanity check: ablation_model with default flags == model.GPT (same param count, same forward given same weights)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import torch
from model import GPT as RefGPT, GPTConfig
from ablation_model import GPT, Cfg

ref = RefGPT(GPTConfig(block_size=256, vocab_size=84, n_layer=6, n_head=6, n_embd=384, dropout=0.0, bias=True)).cuda().eval()
mine = GPT(Cfg(dropout=0.0)).cuda().eval()
print('params ref/mine:', ref.get_num_params(), mine.get_num_params())
# map weights by name (ref has extra none; keys should line up, ignoring the LN param names)
sd_ref = ref.state_dict(); sd_mine = mine.state_dict()
assert set(sd_ref) == set(sd_mine), set(sd_ref) ^ set(sd_mine)
mine.load_state_dict(sd_ref)
x = torch.randint(0, 84, (4, 100)).cuda(); y = torch.randint(0, 84, (4, 100)).cuda()
l1 = ref(x, y)[1].item(); l2 = mine(x, y)[1].item()
print(f'loss ref={l1:.6f} mine={l2:.6f}  |diff|={abs(l1-l2):.2e}')
assert abs(l1 - l2) < 1e-4
print('OK: identical')
