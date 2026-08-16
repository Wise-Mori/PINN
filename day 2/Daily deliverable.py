import torch
torch.manual_seed(0)
torch.set_default_dtype(torch.float64)
print("Day 2 derivatives")
print("u(x,t) = x^2 * exp(-t)")
x = torch.rand(50, 1, requires_grad=True)
t = torch.rand(50, 1, requires_grad=True)
u = x**2 * torch.exp(-t)
u_x = torch.autograd.grad(u, x,grad_outputs=torch.ones_like(u),create_graph=True)[0]
u_t = torch.autograd.grad(u, t,grad_outputs=torch.ones_like(u),create_graph=True)[0]
u_xx = torch.autograd.grad(u_x, x,grad_outputs=torch.ones_like(u_x),create_graph=True)[0]
u_x_closed = 2 * x * torch.exp(-t)
u_t_closed = -x**2 * torch.exp(-t)
u_xx_closed = 2 * torch.exp(-t)
print("calculate error ... ")
u_x_error = torch.abs(u_x - u_x_closed)
u_t_error = torch.abs(u_t - u_t_closed)
u_xx_error = torch.abs(u_xx - u_xx_closed)
print("Autograd or caclulate by hand (manually)")
print(f"max u_x error:  {u_x_error.max().item():.3e}")
print(f"max u_t error:  {u_t_error.max().item():.3e}")
print(f"max u_xx error: {u_xx_error.max().item():.3e}")
assert u_x_error.max() < 1e-10, "Bad gradient: u_x is incorrect"
assert u_xx_error.max() < 1e-10, "Bad gradient: u_xx is incorrect"
print("Assertions passed")
print("error table for u_x : ")
print("h       max error")
print("----------------")
for power in range(1, 8):
    h = 10.0 ** (-power)
    u_plus = (x + h)**2 * torch.exp(-t)
    u_minus = (x - h)**2 * torch.exp(-t)
    u_x_fd = (u_plus - u_minus) / (2 * h)
    fd_error = torch.abs(u_x_fd - u_x)
    max_fd_error = fd_error.max()
    print(f"1e-{power}    {max_fd_error.item():.3e}")
print("Done")