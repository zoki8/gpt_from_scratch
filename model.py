import torch
import torch.nn as nn
from torch.nn import functional as F


class BigramModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets = None):
        logits = self.table(idx)

        if targets is None:
            return logits, None
        
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))
        return logits, loss

if __name__ == "__main__":
    from data import get_batch, tok
    model = BigramModel(tok.vocab_size)
    x, y = get_batch('train', block_size=8, batch_size=4)
    logits,loss = model(x,y)
    print(logits.shape, loss.item())