"""
Char-level dataset built from Gutenberg Shakespeare plays (used because
raw.githubusercontent.com is blocked on Bolt, so tinyshakespeare can't be downloaded).
The plays come from the PyPI sdist `shakespeare==0.6` (shksprdata/texts/*_gut.txt).

usage: python prepare.py /path/to/shksprdata/texts
Writes train.bin / val.bin / meta.pkl next to this file (90/10 split, same as shakespeare_char).
"""
import os, sys, glob, pickle
import numpy as np

src = sys.argv[1]
files = sorted(f for f in glob.glob(os.path.join(src, '*_gut.txt')))  # skip *_gut_f.txt (folio duplicates)
data = '\n'.join(open(f, encoding='latin-1').read() for f in files)
# keep only characters that are plain ASCII to keep the vocab small
data = ''.join(c for c in data if ord(c) < 128)
print(f"{len(files)} plays, {len(data):,} characters")

chars = sorted(set(data))
vocab_size = len(chars)
print("vocab size:", vocab_size)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

n = len(data)
# split by position (plays are alphabetical, so val = last plays, i.e. unseen plays -> honest generalization test)
train_ids = np.array([stoi[c] for c in data[:int(n*0.9)]], dtype=np.uint16)
val_ids = np.array([stoi[c] for c in data[int(n*0.9):]], dtype=np.uint16)
print(f"train {len(train_ids):,} tokens, val {len(val_ids):,} tokens")
d = os.path.dirname(os.path.abspath(__file__))
train_ids.tofile(os.path.join(d, 'train.bin'))
val_ids.tofile(os.path.join(d, 'val.bin'))
with open(os.path.join(d, 'meta.pkl'), 'wb') as f:
    pickle.dump({'vocab_size': vocab_size, 'itos': itos, 'stoi': stoi}, f)
