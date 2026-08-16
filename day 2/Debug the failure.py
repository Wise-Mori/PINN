# Debug the failure
# xt = torch.rand(100, 2) # default requires_grad=False
# u = model(xt)
# du = torch.autograd.grad(u, xt, torch.ones_like(u), create_graph=True)[0]

import torch
import torch.nn as nn
torch.manual_seed(0)
model = nn.Sequential(nn.Linear(2,10),nn.Tanh(),nn.Linear(10,1))
xt = torch.rand(100, 2, requires_grad=True)
u = model(xt)
du = torch.autograd.grad(u, xt,grad_outputs=torch.ones_like(u),create_graph=True)[0]
print("du ")
print(du.shape)
xt = torch.rand(100, 2, requires_grad=True)
u = model(xt)
du = torch.autograd.grad(u, xt,grad_outputs=torch.ones_like(u),create_graph=True)[0]
d2u = torch.autograd.grad(du, xt,grad_outputs=torch.ones_like(du),create_graph=True)[0]
print(d2u)