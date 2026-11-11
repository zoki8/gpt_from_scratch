import sys
import torch
from model import AttentionModel
from tokenizer import CharTokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

checkpoint = torch.load('model.pt', map_location=device)
tok = CharTokenizer(''.join(checkpoint['chars']))
model = AttentionModel(tok.vocab_size, **checkpoint['config']).to(device)
model.load_state_dict(checkpoint['model'])
model.eval()

prompt = sys.argv[1] if len(sys.argv) > 1 else '\n'
length = int(sys.argv[2]) if len(sys.argv) > 2 else 500

unknown = sorted(set(prompt) - set(tok.chars))
if unknown:
    sys.exit(f"These characters are not in model dictionary: {unknown}")

idx = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
with torch.no_grad():
    out = model.generate(idx, max_new_tokens=length)
print(tok.decode(out[0].tolist()))
