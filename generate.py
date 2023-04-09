import torch

def generate(model, model_path, prompt=None):
    model.load_state_dict(torch.load(model_path))
    print(f"This models has {sum(p.numel() for p in model.parameters()) / 1e6} million paramters")
    if prompt:
        if len(prompt) > 8:
            prompt = prompt[:8]
            print(len(prompt))
        elif len(prompt) < 8:
            prompt = "".join([0] * (len(prompt)-8)) + prompt
        prompt = torch.tensor(encode(prompt), dtype=torch.long).view(1,-1)
    else:
        prompt = torch.zeros((1,1), dtype=torch.long)
    print(decode(model.generate(prompt, 300)[0].tolist()))
