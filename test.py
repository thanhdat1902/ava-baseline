import torch

print(torch.cuda.is_initialized())
print(torch.cuda.is_available())


a = torch.tensor([2, 2, 3]).to('cuda:0')
print(a.prod())