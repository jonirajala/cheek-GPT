import torch
from utils import encode
from model import DecoderTransformer

MODEL_PATH = "models/model.pth"
DATA_PATH = "data/rapdataset.txt"
TRAIN_SIZE = 0.9
BLOCK_SIZE = 32
BATCH_SIZE = 32
TRAIN_ITERS = 10000
EVAL_ITERS = 100
LR = 1e-3
N_EMBEDS = 63
NUM_HEADS = 3
N_LAYER = 3
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
        x = torch.stack([data[i:i+BLOCK_SIZE] for i in idx]).to(device)
        y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in idx]).to(device)
        
        return x, y


if __name__ == "__main__":
    with open(DATA_PATH, "r") as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    data = torch.tensor(encode(text, chars), dtype=torch.long)
    model = DecoderTransformer(vocab_size, N_EMBEDS, N_LAYER, BLOCK_SIZE, NUM_HEADS, DROPOUT).to(device)

    print(f"Your data set has {len(data)} characters")
    print(f"Dataset contains: {''.join(chars)} characters")
    print(f"This models has {sum(p.numel() for p in model.parameters()) / 1e6} million paramters")

    train_n = int(TRAIN_SIZE*len(data))
    train_data = data[:train_n]
    test_data = data[train_n:]


    optim = torch.optim.Adam(params=model.parameters(), lr=LR)
    for i in range(0, TRAIN_ITERS):
        x, y = get_batch(training=True)
        optim.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        optim.step()
        if i % 100 == 0:
            eval_loss, train_loss = calc_loss()
            print(f"{i+1}/{TRAIN_ITERS}, Train loss: {train_loss}, Eval loss: {eval_loss}")

    torch.save(model.state_dict(), MODEL_PATH)