# practice 1 - day 2
print("Practice 1 - day 2")
# print("\n")
print("u(x,t) = x²e⁻ᵗ")
# import torch

# x = torch.tensor([[1.0], [2.0], [3.0]], requires_grad=True)
# t = torch.tensor([[0.5], [1.0], [1.5]], requires_grad=True)

# # u(x,t) = x^2 * e^(-t)
# u = x**2 * torch.exp(-t)

# # du/dx
# u_x = torch.autograd.grad(
#     u, x,
#     grad_outputs=torch.ones_like(u),
#     create_graph=True
# )[0]

# # du/dt
# u_t = torch.autograd.grad(
#     u, t,
#     grad_outputs=torch.ones_like(u),
#     create_graph=True
# )[0]

# # d2u/dx2
# u_xx = torch.autograd.grad(
#     u_x, x,
#     grad_outputs=torch.ones_like(u_x),
#     create_graph=True
# )[0]

# # closed form derivatives
# u_x_closed = 2 * x * torch.exp(-t)
# u_t_closed = -x**2 * torch.exp(-t)
# u_xx_closed = 2 * torch.exp(-t)

# print("u:")
# print(u)

# print("\nAutograd u_x:")
# print(u_x)
# print("Closed form u_x:")
# print(u_x_closed)

# print("\nAutograd u_t:")
# print(u_t)
# print("Closed form u_t:")
# print(u_t_closed)

# print("\nAutograd u_xx:")
# print(u_xx)
# print("Closed form u_xx:")
# print(u_xx_closed)