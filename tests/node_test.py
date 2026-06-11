"""Train a Neural ODE to learn CartPole dynamics from rollout data."""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchdiffeq import odeint
from tqdm import tqdm

# Physical constants
CART_MASS = 1.0
POLE_MASS = 0.1
POLE_HALF = 0.5
GRAVITY = 9.81

# Experiment defaults
N_TRAJ = 64 * 512
T_END = 1.0
N_STEPS = 30
U_MAX = 5.0
SEED = 42
STATE_DIM = 4
ACTION_DIM = 1
HIDDEN = 64
N_EPOCHS = 1000
BATCH_SIZE = 64
LR = 1e-2
SHOW_PLOTS = True
BAR_FMT = "{l_bar}{bar} | {n_fmt}/{total_fmt} [{remaining}]"
EVAL_PATH = "imgs/neural_ode_cartpole_eval.png"
LOSS_PATH = "imgs/neural_ode_cartpole_loss.png"


def cartpole_dynamics(t, state, u):
    """state: (B, 4) = [x, xd, th, thd], u: (B, 1) force."""
    xd, th = state[:, 1:2], state[:, 2:3]
    thd = state[:, 3:4]

    sin_th, cos_th = torch.sin(th), torch.cos(th)
    mass = CART_MASS + POLE_MASS

    tmp = (u + POLE_MASS * POLE_HALF * thd**2 * sin_th) / mass
    th_acc = (GRAVITY * sin_th - cos_th * tmp) / (
        POLE_HALF * (4 / 3 - POLE_MASS * cos_th**2 / mass)
    )
    x_acc = tmp - POLE_MASS * POLE_HALF * th_acc * cos_th / mass

    return torch.cat([xd, x_acc, thd, th_acc], dim=-1)


def make_dataset(n_traj=N_TRAJ, T=T_END, n_steps=N_STEPS, u_max=U_MAX, seed=SEED):
    """Roll out true dynamics with manual RK4."""
    torch.manual_seed(seed)
    dt = T / n_steps
    t = torch.linspace(0, T, n_steps + 1)

    # Small random perturbations around upright equilibrium.
    x0 = torch.zeros(n_traj, 4)
    x0[:, 0] = torch.randn(n_traj) * 0.20
    x0[:, 1] = torch.randn(n_traj) * 0.10
    x0[:, 2] = torch.randn(n_traj) * 0.15
    x0[:, 3] = torch.randn(n_traj) * 0.10

    u = (torch.rand(n_traj, 1) * 2 - 1) * u_max

    def f(s):
        return cartpole_dynamics(None, s, u)

    traj = [x0]
    s = x0.clone()
    for _ in tqdm(range(n_steps)):
        k1 = f(s)
        k2 = f(s + 0.5 * dt * k1)
        k3 = f(s + 0.5 * dt * k2)
        k4 = f(s + dt * k3)
        s = s + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        traj.append(s.clone())

    return torch.stack(traj, dim=0), u, t

def rel_l2_loss(pred, target, eps=1e-8) -> torch.Tensor:
    """Per-sample relative L2, averaged over batch."""
    return ((pred - target).norm(dim=(1, 2)) / (target.norm(dim=(1, 2)) + eps)).mean()


class Swish(nn.Module):
    def __init__(self, no=1):
        super().__init__()
        beta = torch.tensor(1.0, dtype=torch.float32) if no == 1 else torch.ones(no, dtype=torch.float32)
        self.beta = nn.Parameter(beta, requires_grad=True)

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class ODEFunc(nn.Module):
    """Small MLP for dx/dt = f(x, u). Set self._u before odeint."""

    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            Swish(),
            # nn.Linear(hidden, hidden),
            # Swish(),
            nn.Linear(hidden, state_dim),
            Swish(),
        )
        self._u = None

    def forward(self, t, x):
        return self.net(torch.cat([x, self._u], dim=-1))


def train(n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, lr=LR):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")

    print("Generating dataset...")
    traj, u_all, t = make_dataset()
    print(f"Dataset : {traj.shape[1]} trajectories, {traj.shape[0]} time steps each")
    traj, u_all, t = traj.to(device), u_all.to(device), t.to(device)

    n = traj.shape[1]
    ode_func = ODEFunc().to(device)
    opt = torch.optim.Adam(ode_func.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    losses = []
    print("Training Neural ODE...")
    for epoch in (bar := tqdm(range(1, n_epochs + 1), bar_format=BAR_FMT)):
        idx = torch.randperm(n, device=device)[:batch_size]
        x0_b, tgt = traj[0, idx], traj[:, idx]

        ode_func._u = u_all[idx]
        pred = odeint(ode_func, x0_b, t, method="rk4")

        # loss = nn.functional.mse_loss(pred, tgt)
        loss = rel_l2_loss(pred, tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

        losses.append(loss.item())
        loss_avg = torch.mean(torch.tensor(losses[-10:]))
        bar.set_description(f"Epoch {epoch:4d} (lr: {sched.get_last_lr()[0]:.2e}) Loss {loss_avg:.2e}")

    return ode_func, losses, traj.cpu(), u_all.cpu(), t.cpu()


def plot_trajectories(ode_func, traj, u_all, t, n_cols=3):
    ode_func.eval()
    labels = ["x [m]", "xd [m/s]", "theta [rad]", "theta_dot [rad/s]"]
    t_np = t.numpy()

    plt.figure(figsize=(5 * n_cols, 10))

    for col in range(n_cols):
        x0 = traj[0, col].unsqueeze(0)
        u = u_all[col].unsqueeze(0)
        ode_func._u = u
        with torch.no_grad():
            pred = odeint(ode_func, x0, t, method="rk4").squeeze(1)

        for row in range(4):
            plt.subplot(4, n_cols, row * n_cols + col + 1)
            plt.plot(t_np, traj[:, col, row].numpy(), lw=2, label="True")
            plt.plot(t_np, pred[:, row].numpy(), "--", lw=2, label="Neural ODE")
            if col == 0: plt.ylabel(labels[row])
            if row == 0: plt.title(f"Trajectory {col + 1} (u = {u_all[col].item():.1f} N)"); plt.legend(fontsize=8)
            if row == 3: plt.xlabel("Time [s]")

    plt.suptitle("Neural ODE CartPole - True vs Predicted", fontsize=13)
    plt.tight_layout()
    plt.savefig(EVAL_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved -> {EVAL_PATH}")
    if SHOW_PLOTS: plt.show()


def plot_loss(losses):
    plt.figure(figsize=(7, 3))
    plt.semilogy(losses)
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss (log scale)")
    plt.title("Training loss")
    plt.tight_layout()
    plt.savefig(LOSS_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved -> {LOSS_PATH}")
    if SHOW_PLOTS: plt.show()


if __name__ == "__main__":
    ode_func, losses, traj, u_all, t = train()
    plot_loss(losses)
    plot_trajectories(ode_func, traj, u_all, t)
