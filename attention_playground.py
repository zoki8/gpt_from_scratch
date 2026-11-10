import torch

x = torch.tensor([[1., 2.],
                  [3., 4.],
                  [5., 6.],
                  [7., 8.]])
T, C = x.shape

# Način 1: petlja
out1 = torch.zeros(T, C)
for t in range(T):
    out1[t] = x[:t+1].mean(dim=0)
print(out1)

# Način 2: množenje matrica
wei = torch.tril(torch.ones(T, T))
print(wei)
# TODO 2: podeli svaki red wei zbirom tog reda
wei = wei/wei.sum(dim = 1, keepdim = True)
print(wei)
out2 = wei @ x
print(out2)

print(torch.allclose(out1, out2))
