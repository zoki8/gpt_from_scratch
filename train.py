import torch
from data import get_batch, tok
from model import BigramModel


model = BigramModel(tok.vocab_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

for step in range(3000):
    x, y = get_batch('train', block_size=8, batch_size=4)
    logits, loss = model(x,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 300 == 0:
        print(step, loss.item())

start = torch.zeros((1, 1), dtype=torch.long)
print(tok.decode(model.generate(start, max_new_tokens=300)[0].tolist()))