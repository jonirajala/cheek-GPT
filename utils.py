def encode(string, chars):
    stoi = {ch:i for i, ch in enumerate(chars)}
    ids = [stoi[char] for char in string]
    return ids

def decode(ids, chars):
    itos = {i:ch for i, ch in enumerate(chars)}
    chars_list = [itos[id] for id in ids]
    return "".join(chars_list)
