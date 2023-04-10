import torch
from model import CheekGPT
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

DATA_PATH = 'data/bpe_level_rap'
MODEL_PATH = "models/bpe_level_models/model.pth"
BLOCK_SIZE = 128
BATCH_SIZE = 32
TRAIN_ITERS = 3500
EVAL_ITERS = 100
LR = 1e-3
N_EMBEDS = 160
NUM_HEADS = 8
N_LAYER = 8
DROPOUT = 0.2
device = 'cuda' if torch.cuda.is_available() else 'cpu'


@torch.no_grad()
def calc_loss():
    model.eval()
    losses = []
    for i in range(2):
        cum_loss = 0
        for _ in range(EVAL_ITERS):
            x, y = get_batch(training=i)
            _, loss = model(x, y)
            cum_loss += loss.item()
        losses.append(cum_loss/EVAL_ITERS)

    model.train()
    return losses[0], losses[1]


def get_batch(training=True):
        data = train_data if training else test_data
        idx = torch.randint(0,len(data)-BLOCK_SIZE, (BATCH_SIZE,))
        x = torch.stack([torch.from_numpy((data[i:i+BLOCK_SIZE]).astype(np.int64)) for i in idx]).to(device)
        y = torch.stack([torch.from_numpy((data[i+1:i+BLOCK_SIZE+1]).astype(np.int64)) for i in idx]).to(device)
        
        return x, y


if __name__ == "__main__":
    train_data = np.memmap(os.path.join(DATA_PATH, 'train.bin'), dtype=np.uint16, mode='r')
    test_data = np.memmap(os.path.join(DATA_PATH, 'test.bin'), dtype=np.uint16, mode='r')

    files = [f for f in os.listdir(DATA_PATH) if os.path.isfile(os.path.join(DATA_PATH, f))]
    if 'meta.pkl' in files:
        with open(os.path.join(DATA_PATH, 'meta.pkl'), 'rb') as f:
            meta = pickle.load(f)
        vocab_size = meta['vocab_size']
    else:
        # vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)
        vocab_size =  50304
    
    print(f"found vocab size of {vocab_size}")

    model = CheekGPT(vocab_size, N_EMBEDS, N_LAYER, BLOCK_SIZE, NUM_HEADS, DROPOUT).to(device)

    print(f"Your data has {len(train_data) + len(test_data)} tokens")
    print(f"This models has {sum(p.numel() for p in model.parameters()) / 1e6} million paramters")


    optim = torch.optim.Adam(params=model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=700, gamma=0.1)
    model.train()
    losses = {"train": [], "test": []}
    for i in range(0, TRAIN_ITERS):
        x, y = get_batch(training=True)
        optim.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        optim.step()
        if i % 100 == 0:
            test_loss, train_loss = calc_loss()
            losses["train"].append(train_loss)
            losses["test"].append(test_loss)
            print(f"{i+1}/{TRAIN_ITERS}, Train loss: {train_loss}, Eval loss: {test_loss}")
        scheduler.step()


    torch.save(model.state_dict(), MODEL_PATH)

    plt.plot(np.arange(0, TRAIN_ITERS//100, losses["train"]))
    plt.plot(np.arange(0, TRAIN_ITERS//100), losses["test"])    
