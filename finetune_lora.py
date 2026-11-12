import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import torch
from torch.nn import functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from lora import apply_lora, lora_state_dict

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 4
block_size = 256
max_steps = 500
eval_interval = 100
eval_iters = 20
learning_rate = 5e-4
model_name = 'gpt2-medium'

tok = GPT2TokenizerFast.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name, dtype=torch.bfloat16)
model = apply_lora(model, r=8, alpha=16)
model = model.to(device)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"uči se {trainable} od {total} parametara ({100 * trainable / total:.2f}%)")

text = open('data/input.txt', encoding='utf-8').read()
data = torch.tensor(tok.encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]


def get_batch(split):
    source = train_data if split == 'train' else val_data
    starts = torch.randint(len(source) - block_size, (batch_size,))
    x = torch.stack([source[i:i + block_size] for i in starts])
    y = torch.stack([source[i + 1:i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)


def compute_loss(x, y):
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=(device == 'cuda')):
        logits = model(x).logits.float()      # (B, T, 50257)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1))

    return loss


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            losses[k] = compute_loss(x, y).item()
        out[split] = losses.mean().item()
    model.train()
    return out


def sample(prompt, length=150):
    model.eval()
    idx = torch.tensor([tok.encode(prompt)], device=device)
    with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=(device == 'cuda')):
        out = model.generate(idx, max_new_tokens=length, do_sample=True, temperature=0.8,
                             top_k=50, pad_token_id=tok.eos_token_id)
    model.train()
    return tok.decode(out[0].tolist())


chars_per_token = len(text) / len(data)
print(f"Šekspir: {len(data)} GPT-2 tokena, {chars_per_token:.2f} karaktera po tokenu")
print("=== GPT-2 PRE dotreniravanja:")
print(sample("ROMEO:"))

optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
for step in range(max_steps):
    if step % eval_interval == 0:
        l = estimate_loss()
        print(f"{step}: train {l['train']:.3f}, val {l['val']:.3f}  (po karakteru: val {l['val'] / chars_per_token:.3f})")
    x, y = get_batch('train')
    loss = compute_loss(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

l = estimate_loss()
print(f"{max_steps}: train {l['train']:.3f}, val {l['val']:.3f}  (po karakteru: val {l['val'] / chars_per_token:.3f})")
print("=== GPT-2 POSLE dotreniravanja:")
print(sample("ROMEO:"))

torch.save(lora_state_dict(model), 'lora_medium.pt')
print("LoRA sačuvana u lora_medium.pt")
