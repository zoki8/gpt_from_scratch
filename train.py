import torch
from data import get_batch, tok, text, data
from model import AttentionModel

batch_size = 64
block_size = 128
max_steps = 5000
eval_interval = 500
eval_iters = 100
n_embd = 256
n_head = 4
n_layer = 6
dropout = 0.2
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = AttentionModel(tok.vocab_size, n_embd, block_size, n_head, n_layer, dropout).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

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

def save_checkpoint(step, val_loss):
    checkpoint = {
        'model': model.state_dict(),
        'config': dict(n_embd=n_embd, block_size=block_size, n_head=n_head, n_layer=n_layer, dropout=dropout),
        'tokenizer': tok.to_dict(),
        'val_loss': val_loss,
        'step': step,
    }
    torch.save(checkpoint, 'model.pt')


best_val = float('inf')
best_step = None


def evaluate(step):
    global best_val, best_step
    l = estimate_loss()
    val = l['val'].item()
    mark = ''
    if val < best_val:
        best_val, best_step = val, step
        save_checkpoint(step, val)
        mark = '  <- najbolji do sada, sačuvan'
    print(f"{step}: train {l['train']:.3f}, val {val:.3f}{mark}")


for step in range(max_steps):
    if step % eval_interval == 0:
        evaluate(step)
    x, y = get_batch('train', block_size, batch_size, device= device)
    logits, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

evaluate(max_steps)
print(f"Najbolji model: korak {best_step}, val {best_val:.3f} "
      f"({best_val * len(data) / len(text):.3f} po karakteru, {len(text) / len(data):.2f} karaktera po tokenu), sačuvan u model.pt")

model.load_state_dict(torch.load('model.pt', map_location=device)['model'])
model.eval()
start = torch.zeros((1, 1), dtype=torch.long, device = device)
print(tok.decode(model.generate(start, max_new_tokens=300)[0].tolist()))