# practice 1 - day 3
import torch
import torch.nn as nn
import torch.optim as optim
torch.manual_seed(0)
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, 32),nn.Tanh(),nn.Linear(32, 32),nn.Tanh(),nn.Linear(32, 1) )
    def forward(self, t):
        return self.net(t)
def loss_terms(model, t):
    t.requires_grad_(True)
    theta = model(t)
    theta_t = torch.autograd.grad(theta, t,grad_outputs=torch.ones_like(theta),create_graph=True)[0]
    residual = theta_t + theta
    loss_f = torch.mean(residual**2)
    t0 = torch.zeros(1, 1)
    theta0 = model(t0)
    loss_b = torch.mean((theta0 - 1.0)**2)
    return loss_f, loss_b
def train_pinn(lambda_f, lambda_b, epochs=3000):
    model = MLP()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        optimizer.zero_grad()
        t = 2 * torch.rand(100, 1)
        loss_f, loss_b = loss_terms(model, t)
        loss = lambda_f * loss_f + lambda_b * loss_b
        loss.backward()
        optimizer.step()
        if epoch % 500 == 0:
            print(
                f"epoch {epoch:4d} | "
                f"loss = {loss.item():.3e} | "
                f"loss_f = {loss_f.item():.3e} | "
                f"loss_b = {loss_b.item():.3e}"
            )
    return model
def validate(model):
    t_test = torch.linspace(0, 2, 100).reshape(-1, 1)
    theta_pred = model(t_test).detach()
    theta_exact = torch.exp(-t_test)
    mse = torch.mean((theta_pred - theta_exact)**2)
    theta0_pred = model(torch.zeros(1, 1)).item()
    return mse.item(), theta0_pred
settings = [(1, 1),(1, 1e-4),(1, 100)]
for lambda_f, lambda_b in settings:
    print("====================================")
    print(f"Training with lambda_f = {lambda_f}, lambda_b = {lambda_b}")
    print("====================================")
    model = train_pinn(lambda_f, lambda_b)
    mse, theta0_pred = validate(model)
    print("\nValidation:")
    print(f"MSE VS exp(-t): {mse:.3e}")
    print(f"theta(0) prediction: {theta0_pred:.6f}")