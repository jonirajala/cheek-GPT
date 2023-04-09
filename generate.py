import torch
from model import CheekGPT
import os
import pickle
import tiktoken
from train import N_EMBEDS, N_LAYER, BLOCK_SIZE, NUM_HEADS, DROPOUT, MODEL_PATH, DATA_PATH



if __name__ == "__main__":
    files = [f for f in os.listdir(DATA_PATH) if os.path.isfile(os.path.join(DATA_PATH, f))]
    if 'meta.pkl' in files:
        with open(os.path.join(DATA_PATH, 'meta.pkl'), 'rb') as f:
            meta = pickle.load(f)

        stoi, itos = meta['stoi'], meta['itos']
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
        vocab_size = meta['vocab_size']

    else:
        vocab_size =  50304
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)


    model = CheekGPT(vocab_size, N_EMBEDS, N_LAYER, BLOCK_SIZE, NUM_HEADS, DROPOUT)
    model.load_state_dict(torch.load(MODEL_PATH))
    print(f"This models has {sum(p.numel() for p in model.parameters()) / 1e6} million paramters")
    
    prompt = torch.zeros((1,1), dtype=torch.long)

    
    print(decode(model.generate(prompt, 300)[0].tolist()))
