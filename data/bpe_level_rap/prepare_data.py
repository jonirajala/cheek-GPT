import numpy as np
import os
import tiktoken

TRAIN_SIZE = 0.9
DATA_PATH = "../raw_data/raw_rapdataset.txt"


with open(DATA_PATH, "r") as f:
    text = f.read()

encoder = tiktoken.get_encoding("gpt2")
data = encoder.encode_ordinary(text)

print(f"Your data has {len(data)} tokens")

train_n = int(TRAIN_SIZE*len(data))
train_data = data[:train_n]
test_data = data[train_n:]

train_data = np.array(train_data, dtype=np.uint16)
test_data = np.array(test_data, dtype=np.uint16)

train_data.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
test_data.tofile(os.path.join(os.path.dirname(__file__), 'test.bin'))
