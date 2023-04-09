import torch
from utils import decode, encode
from model import DecoderTransformer
from train import N_EMBEDS, N_LAYER, BLOCK_SIZE, NUM_HEADS, DROPOUT, MODEL_PATH, DATA_PATH

with open(DATA_PATH, "r") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

if __name__ == "__main__":
    model = DecoderTransformer(vocab_size, N_EMBEDS, N_LAYER, BLOCK_SIZE, NUM_HEADS, DROPOUT)
    model.load_state_dict(torch.load(MODEL_PATH))
    print(f"This models has {sum(p.numel() for p in model.parameters()) / 1e6} million paramters")
    
    prompt = torch.zeros((1,1), dtype=torch.long)
    print(decode(model.generate(prompt, 300)[0].tolist(), chars))
