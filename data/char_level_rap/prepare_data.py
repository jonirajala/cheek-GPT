import numpy as np
import os
import torch
import pickle

TRAIN_SIZE = 0.9
DATA_PATH = "../raw_data/raw_rapdataset.txt"


with open(DATA_PATH, "r") as f:
    text = f.read()

chars = sorted(list(set(text)))

stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}

def encode(string, chars):
    ids = [stoi[char] for char in string]
    return ids

def decode(ids, chars):
    chars_list = [itos[id] for id in ids]
    return "".join(chars_list)

data = np.array(encode(text, chars), dtype=np.uint16)

vocab_size = len(chars)

train_n = int(TRAIN_SIZE*len(data))
train_data = data[:train_n]
test_data = data[train_n:]

print(f"Your data has {len(data)} tokens")

train_data.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
test_data.tofile(os.path.join(os.path.dirname(__file__), 'test.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)
