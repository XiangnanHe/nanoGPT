"""Tiny autograder used by the exercise cells in nanogpt_tutorial.ipynb"""
import torch

def check(label, fn):
    try:
        fn()
        print(f'✅ {label}')
    except NotImplementedError:
        print(f'⏳ {label}: not implemented yet (fill in the TODOs above)')
    except AssertionError as e:
        print(f'❌ {label}: {e}')
    except Exception as e:
        print(f'❌ {label}: {type(e).__name__}: {e}')

def load_ref_weights(mine, ref):
    """Copy the reference module's weights into yours. Fails with a readable message if your parameter NAMES/SHAPES differ."""
    res = mine.load_state_dict(ref.state_dict(), strict=False)
    assert not res.missing_keys, f'your module is missing parameters named {res.missing_keys} (names must match model.py)'

def close(a, b, what='output', atol=1e-5):
    assert a.shape == b.shape, f'{what}: shape {tuple(a.shape)} but expected {tuple(b.shape)}'
    err = (a - b).abs().max().item()
    assert err < atol, f'{what}: max abs difference to model.py = {err:.2e} (should be < {atol})'
