import torch
from data import get_batch, tok
from model import AttentionModel

batch_size = 64
block_size = 64
max_steps = 5000
eval_interval = 500
eval_iters = 100
n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.1
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = AttentionModel(tok.vocab_size, n_embd, block_size, n_head, n_layer, dropout).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split, block_size, batch_size, device=device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

for step in range(max_steps):
    if step % eval_interval == 0:
        l = estimate_loss()
        print(f"{step}: train {l['train']:.3f}, val {l['val']:.3f}")
    x, y = get_batch('train', block_size, batch_size, device= device)
    logits, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

l = estimate_loss()
print(f"{max_steps}: train {l['train']:.3f}, val {l['val']:.3f}")

checkpoint = {
    'model': model.state_dict(),
    'config': dict(n_embd=n_embd, block_size=block_size, n_head=n_head, n_layer=n_layer, dropout=dropout),
    'chars': tok.chars,
    'val_loss': l['val'].item(),
}
torch.save(checkpoint, 'model.pt')
print("Model sačuvan u model.pt")

model.eval()
start = torch.zeros((1, 1), dtype=torch.long, device = device)
print(tok.decode(model.generate(start, max_new_tokens=300)[0].tolist()))