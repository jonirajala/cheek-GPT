import torch
from utils import decode
from model import CheekGPT
from train import N_EMBEDS, N_LAYER, BLOCK_SIZE, NUM_HEADS, DROPOUT, MODEL_PATH, DATA_PATH

if __name__ == "__main__":
    with open(DATA_PATH, "r") as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    model = CheekGPT(vocab_size, N_EMBEDS, N_LAYER, BLOCK_SIZE, NUM_HEADS, DROPOUT)
    model.load_state_dict(torch.load(MODEL_PATH))
    print(f"This models has {sum(p.numel() for p in model.parameters()) / 1e6} million paramters")
    
    prompt = torch.zeros((1,1), dtype=torch.long)
    print(decode(model.generate(prompt, 300)[0].tolist(), chars))
