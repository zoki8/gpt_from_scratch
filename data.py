import json
import os
import torch
from tokenizer import CharTokenizer, BPETokenizer, load_tokenizer

TOKENIZER = 'bpe'          # 'char' or 'bpe'
BPE_VOCAB_SIZE = 512
BPE_PATH = 'bpe.json'

text = open('data/input.txt', encoding = 'utf-8').read()
if TOKENIZER == 'bpe':
    if os.path.exists(BPE_PATH):
        tok = load_tokenizer(json.load(open(BPE_PATH)))
    else:
        tok = BPETokenizer.train(text, BPE_VOCAB_SIZE)
        json.dump(tok.to_dict(), open(BPE_PATH, 'w'))
else:
    tok = CharTokenizer(text)

data = torch.tensor(tok.encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split: str, block_size: int, batch_size: int, device: str = 'cpu') -> tuple[torch.Tensor, torch.Tensor]:
    source = train_data if split == 'train' else val_data
    starts = torch.randint(len(source) - block_size, (batch_size,))
    x = torch.stack([source[i:i + block_size] for i in starts])
    y = torch.stack([source[i + 1:i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)


if __name__ == "__main__":
    print(len(train_data), len(val_data))
    x, y = get_batch('train', block_size=8, batch_size=4)
    print(x.shape, y.shape)
    print(tok.decode(x[0].tolist()))
    print(tok.decode(y[0].tolist()))
    assert (x[:, 1:] == y[:, :-1]).all()
