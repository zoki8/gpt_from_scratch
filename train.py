import torch
from data import get_batch, tok
from model import BigramModel

batch_size = 32
block_size = 8
max_steps = 3000
eval_interval = 300
eval_iters = 200

model = BigramModel(tok.vocab_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split, block_size, batch_size)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

for step in range(max_steps):
    if step % eval_interval == 0:
        l = estimate_loss()
        print(f"{step}: train {l['train']:.3f}, {l['val']:.3f}")
    x, y = get_batch('train', block_size, batch_size)
    logits, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


start = torch.zeros((1, 1), dtype=torch.long)
print(tok.decode(model.generate(start, max_new_tokens=300)[0].tolist()))