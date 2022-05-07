import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
a = torch.randn(300, 300).to(device)
a = a**2
print(a)
print(device)