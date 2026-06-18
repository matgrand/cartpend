"""FNO CartPole: learn the solution operator (x0, u) -> full trajectory."""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Physical constants
CART_MASS, POLE_MASS, POLE_HALF, GRAVITY = 1.0, 0.1, 0.5, 9.81

# Dataset
N_TRAJ  = 64 * 512 * 10
T_END   = 1.0 *2
N_STEPS = 130 *2      # trajectory has N_STEPS+1 time points
U_MAX   = 5.0
SEED    = 42
STATE_DIM, ACT_DIM = 4, 1
IN_CH   = STATE_DIM + ACT_DIM   # 5

# FNO architecture
N_MODES    = 12      # < N_STEPS/2 to stay in Nyquist limit, avoids aliased nonlinear modes
HIDDEN     = 64
N_LAYERS   = 4

# Training
N_EPOCHS   = 10000
BATCH_SIZE = 64
LR         = 1e-2
SHOW_PLOTS = True
EVAL_PATH  = "imgs/fno_cartpole_eval.png"
LOSS_PATH  = "imgs/fno_cartpole_loss.png"
BAR_FMT    = "{l_bar}{bar} | {n_fmt}/{total_fmt} [{remaining}]"


# ── dynamics ──────────────────────────────────────────────────────────────────
def cartpole_dynamics(state, u):
    xd, th, thd = state[:, 1:2], state[:, 2:3], state[:, 3:4]
    sin_th, cos_th = torch.sin(th), torch.cos(th)
    mass   = CART_MASS + POLE_MASS
    tmp    = (u + POLE_MASS * POLE_HALF * thd**2 * sin_th) / mass
    th_acc = (GRAVITY * sin_th - cos_th * tmp) / (POLE_HALF * (4/3 - POLE_MASS * cos_th**2 / mass))
    x_acc  = tmp - POLE_MASS * POLE_HALF * th_acc * cos_th / mass
    return torch.cat([xd, x_acc, thd, th_acc], dim=-1)


def make_dataset():
    torch.manual_seed(SEED)
    dt = T_END / N_STEPS
    t  = torch.linspace(0, T_END, N_STEPS + 1)
    x0 = torch.zeros(N_TRAJ, 4)
    x0[:, 0] = torch.randn(N_TRAJ) * 0.20 *10 
    x0[:, 1] = torch.randn(N_TRAJ) * 0.10 *10
    x0[:, 2] = torch.randn(N_TRAJ) * 0.15 *10
    x0[:, 3] = torch.randn(N_TRAJ) * 0.10 *10
    u = (torch.rand(N_TRAJ, 1) * 2 - 1) * U_MAX

    traj = [x0]; s = x0.clone()
    for _ in tqdm(range(N_STEPS), desc="Dataset"):
        k1 = cartpole_dynamics(s, u)
        k2 = cartpole_dynamics(s + 0.5*dt*k1, u)
        k3 = cartpole_dynamics(s + 0.5*dt*k2, u)
        k4 = cartpole_dynamics(s + dt*k3, u)
        s  = s + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        traj.append(s.clone())

    traj  = torch.stack(traj, dim=0)                          # (T+1, N, 4)
    u_rep = u.unsqueeze(0).expand(N_STEPS+1, -1, -1)         # (T+1, N, 1)
    return torch.cat([traj, u_rep], dim=-1), t                # (T+1, N, 5), (T+1,)


def rel_l2_loss(pred, target, eps=1e-8) -> torch.Tensor:
    """Per-sample relative L2, averaged over batch."""
    return ((pred - target).norm(dim=(1, 2)) / (target.norm(dim=(1, 2)) + eps)).mean()


def split(data):
    """data (T+1, N, 5) -> x_inp (N, 5), y_tgt (N, 4, T+1)."""
    x_inp = data[0]                                    # (N, 5) -- [x0, u]
    y_tgt = data[:, :, :STATE_DIM].permute(1, 2, 0)   # (N, 4, T+1) -- full trajectory
    return x_inp, y_tgt


# ── from-scratch FNO ──────────────────────────────────────────────────────────
class SpectralConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, n_modes):
        super().__init__()
        self.n_modes = n_modes
        scale = (2 / (in_ch + out_ch)) ** 0.5
        self.weight = nn.Parameter(scale * torch.randn(in_ch, out_ch, n_modes, dtype=torch.cfloat))

    def forward(self, x):  # (B, C_in, T)
        B, _, T = x.shape
        x_ft   = torch.fft.rfft(x, norm="forward")
        out_ft = torch.zeros(B, self.weight.shape[1], x_ft.shape[-1], dtype=torch.cfloat, device=x.device)
        m = min(self.n_modes, x_ft.shape[-1])
        out_ft[:, :, :m] = torch.einsum("bci,coi->boi", x_ft[:, :, :m], self.weight[:, :, :m])
        return torch.fft.irfft(out_ft, n=T, norm="forward")


class FNO1d(nn.Module):
    def __init__(self):
        super().__init__()
        self.lift_inp  = nn.Linear(IN_CH, HIDDEN)              # [x0, u] -> hidden bias
        self.lift_grid = nn.Conv1d(1, HIDDEN, 1)               # time grid -> hidden
        self.convs = nn.ModuleList(SpectralConv1d(HIDDEN, HIDDEN, N_MODES) for _ in range(N_LAYERS))
        self.skips = nn.ModuleList(nn.Conv1d(HIDDEN, HIDDEN, 1) for _ in range(N_LAYERS))
        self.proj  = nn.Sequential(nn.Conv1d(HIDDEN, HIDDEN, 1), nn.GELU(), nn.Conv1d(HIDDEN, STATE_DIM, 1))

    def forward(self, x):  # x: (B, IN_CH)
        B, T = x.shape[0], N_STEPS + 1
        grid = torch.linspace(0, 1, T, device=x.device).view(1, 1, T).expand(B, 1, T)
        h = self.lift_inp(x).unsqueeze(-1) + self.lift_grid(grid)   # (B, HIDDEN, T)
        for i, (conv, skip) in enumerate(zip(self.convs, self.skips)):
            h = conv(h) + skip(h)
            if i < N_LAYERS - 1: h = F.gelu(h)
        return self.proj(h)

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(IN_CH, HIDDEN), nn.GELU(),
            *[l for _ in range(N_LAYERS) for l in (nn.Linear(HIDDEN, HIDDEN), nn.GELU())],
            nn.Linear(HIDDEN, STATE_DIM * (N_STEPS + 1)),
        )

    def forward(self, x):  # (B, IN_CH)
        return self.net(x).view(x.shape[0], STATE_DIM, N_STEPS + 1)

# ── training ──────────────────────────────────────────────────────────────────
def train(data, model, name=""):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = model.to(device)
    x_inp, y_tgt = split(data)
    x_inp, y_tgt = x_inp.to(device), y_tgt.to(device)
    n     = x_inp.shape[0]
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS)

    losses = []
    for epoch in (bar := tqdm(range(1, N_EPOCHS+1), bar_format=BAR_FMT, desc=name)):
        idx  = torch.randperm(n, device=device)[:BATCH_SIZE]
        loss = rel_l2_loss(model(x_inp[idx]), y_tgt[idx])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        losses.append(loss.item())
        if epoch % 10 == 0: bar.set_description(f"{name} {epoch:4d} | RelL2 {torch.tensor(losses[-10:]).mean():.2e}")
    return losses


# ── plotting ──────────────────────────────────────────────────────────────────
def plot_loss(losses_dict):
    plt.figure(figsize=(7, 3))
    for name, losses in losses_dict.items(): plt.semilogy(losses, label=name)
    plt.xlabel("Epoch"); plt.ylabel("Relative L2"); plt.title("Training loss"); plt.legend()
    plt.tight_layout(); plt.savefig(LOSS_PATH, dpi=150, bbox_inches="tight"); print(f"Saved -> {LOSS_PATH}")


def plot_eval(models_dict, data, t, n_cols=3):
    labels = ["x [m]", "xd [m/s]", "θ [rad]", "θd [rad/s]"]
    x_inp, y_tgt = split(data)

    plt.figure(figsize=(5*n_cols, 10))
    for col in range(n_cols):
        true = y_tgt[col]
        for row in range(STATE_DIM):
            plt.subplot(STATE_DIM, n_cols, row*n_cols + col + 1)
            plt.plot(t, true[row], lw=2, label="True")
            for name, model in models_dict.items():
                model.eval(); device = next(model.parameters()).device
                with torch.no_grad(): pred = model(x_inp[col:col+1].to(device)).squeeze(0).cpu()
                plt.plot(t, pred[row], "--", lw=2, label=name)
            if col == 0: plt.ylabel(labels[row])
            if row == 0: plt.title(f"Traj {col+1}"); plt.legend(fontsize=8)
            if row == STATE_DIM-1: plt.xlabel("Time [s]")

    plt.suptitle("CartPole — True vs Predicted", fontsize=13)
    plt.tight_layout(); plt.savefig(EVAL_PATH, dpi=150, bbox_inches="tight"); print(f"Saved -> {EVAL_PATH}")


if __name__ == "__main__":
    print("Generating dataset...")
    data, t = make_dataset()
    print(f"Dataset: {data.shape[1]} trajectories, {data.shape[0]} time steps, {data.shape[2]} channels")

    fno, mlp = FNO1d(), SimpleNet()
    print(f"FNO1d params:    {sum(p.numel() for p in fno.parameters()):,}")
    print(f"SimpleNet params:{sum(p.numel() for p in mlp.parameters()):,}")

    losses_fno = train(data, fno, "FNO")
    losses_mlp = train(data, mlp, "MLP")

    plot_loss({"FNO": losses_fno, "MLP": losses_mlp})
    plot_eval({"FNO": fno, "MLP": mlp}, data.cpu(), t)
    if SHOW_PLOTS: plt.show()