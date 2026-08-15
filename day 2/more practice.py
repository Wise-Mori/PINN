# more practice
import torch
torch.manual_seed(0)
x = torch.rand(10, 1, requires_grad=True)
t = torch.rand(10, 1, requires_grad=True)
print("u(x,t) = x^3 + 2*t")
u = x**3 + 2*t
print(f"x:{x}")
print(f"t:{t}")
print(f"u:{u}")
#...............................................
u_x = torch.autograd.grad(
    u, x,
    grad_outputs=torch.ones_like(u),
    create_graph=True
)[0]
print(f"u_x: {u_x}")
#...............................................
u_t = torch.autograd.grad(
    u, t,
    grad_outputs=torch.ones_like(u),
    create_graph=True
)[0]
print(f"u_t: {u_t}")
#................................................
u_xx = torch.autograd.grad(
    u_x, x,
    grad_outputs=torch.ones_like(u_x),
    create_graph=True
)[0]
print(f"u_xx: {u_xx}")
#................................................
u_x_closed = 3 * x**2
u_t_closed = 2 * torch.ones_like(t)
u_xx_closed = 6 * x
print(f"u_x closed form: {u_x_closed}")
print(f"u_t closed form: {u_t_closed}")
print(f"u_xx closed form: {u_xx_closed}")