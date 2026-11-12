import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):

    def __init__(self, base: nn.Module, in_features: int, out_features: int, r: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.A = nn.Parameter(torch.randn(r, in_features) / math.sqrt(in_features))
        self.B = nn.Parameter(torch.zeros(out_features, r))

        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale



def apply_lora(model: nn.Module, r: int = 8, alpha: int = 16, targets=('c_attn',)) -> nn.Module:
    """Zamrzne ceo model i svaki sloj čije je ime u `targets` zameni sa LoRALinear."""
    for p in model.parameters():
        p.requires_grad = False
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if name in targets:
                if isinstance(child, nn.Linear):
                    in_f, out_f = child.in_features, child.out_features
                else:  # GPT-2 koristi Conv1D: težina je oblika (ulaz, izlaz)
                    in_f, out_f = child.weight.shape
                setattr(module, name, LoRALinear(child, in_f, out_f, r, alpha))
    return model


def lora_state_dict(model: nn.Module) -> dict:
    """Samo naučeni LoRA parametri, to je sve što treba sačuvati."""
    return {k: v for k, v in model.state_dict().items() if k.endswith('.A') or k.endswith('.B')}


if __name__ == "__main__":
    torch.manual_seed(0)
    base = nn.Linear(16, 32)
    lora = LoRALinear(nn.Linear(16, 32), 16, 32, r=4)
    lora.base.load_state_dict(base.state_dict())
    x = torch.randn(5, 16)
    assert lora.B.shape == (32, 4) and (lora.B == 0).all(), "B treba da bude nula-matrica oblika (out, r)"
    assert torch.allclose(lora(x), base(x)), "na početku LoRA sloj mora da radi isto kao originalni"
    with torch.no_grad():
        lora.B.fill_(0.1)
    assert not torch.allclose(lora(x), base(x)), "kad B nije nula, izlaz mora da se promeni"
    trainable = [n for n, p in lora.named_parameters() if p.requires_grad]
    assert trainable == ['A', 'B'], f"uče se samo A i B, a ne {trainable}"
    print("LoRALinear: OK")
