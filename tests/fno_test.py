"""FNO CartPole: learn the solution operator (x0, u) -> full trajectory."""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

USE_LIBRARY = False  # True: neuralop FNO | False: from-scratch FNO

# Physical constants
CART_MASS, POLE_MASS, POLE_HALF, GRAVITY = 1.0, 0.1, 0.5, 9.81

# Dataset
N_TRAJ  = 64 * 512
T_END   = 1.0
N_STEPS = 130      # trajectory has N_STEPS+1 time points
U_MAX   = 5.0
SEED    = 42
STATE_DIM, ACT_DIM = 4, 1
IN_CH   = STATE_DIM + ACT_DIM   # 5

# FNO architecture
N_MODES    = 12      # < N_STEPS/2 to stay in Nyquist limit, avoids aliased nonlinear modes
HIDDEN     = 64
N_LAYERS   = 4
DOMAIN_PAD = 0.2   # fractional extra time steps added via FC-Legendre in make_dataset
FC_DEGREE  = 5      # Legendre matching points on each boundary

# Training
N_EPOCHS   = 1000
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
    mass = CART_MASS + POLE_MASS
    tmp    = (u + POLE_MASS * POLE_HALF * thd**2 * sin_th) / mass
    th_acc = (GRAVITY * sin_th - cos_th * tmp) / (POLE_HALF * (4/3 - POLE_MASS * cos_th**2 / mass))
    x_acc  = tmp - POLE_MASS * POLE_HALF * th_acc * cos_th / mass
    return torch.cat([xd, x_acc, thd, th_acc], dim=-1)


def make_dataset():
    from neuralop.layers.fourier_continuation import FCLegendre
    torch.manual_seed(SEED)
    dt = T_END / N_STEPS
    t  = torch.linspace(0, T_END, N_STEPS + 1)
    x0 = torch.zeros(N_TRAJ, 4)
    x0[:, 0] = torch.randn(N_TRAJ) * 0.20
    x0[:, 1] = torch.randn(N_TRAJ) * 0.10
    x0[:, 2] = torch.randn(N_TRAJ) * 0.15
    x0[:, 3] = torch.randn(N_TRAJ) * 0.10
    u = (torch.rand(N_TRAJ, 1) * 2 - 1) * U_MAX

    traj = [x0]; s = x0.clone()
    for _ in tqdm(range(N_STEPS), desc="Dataset"):
        k1 = cartpole_dynamics(s, u)
        k2 = cartpole_dynamics(s + 0.5*dt*k1, u)
        k3 = cartpole_dynamics(s + 0.5*dt*k2, u)
        k4 = cartpole_dynamics(s + dt*k3, u)
        s  = s + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        traj.append(s.clone())

    traj   = torch.stack(traj, dim=0)                           # (T+1, N, 4)
    x0_rep = x0.unsqueeze(-1).expand(-1, -1, N_STEPS+1)        # (N, 4, T+1)
    u_rep  = u.unsqueeze(-1).expand(-1, -1, N_STEPS+1)         # (N, 1, T+1)
    x_data = torch.cat([x0_rep, u_rep], dim=1)                 # (N, 5, T+1)
    y_data = traj.permute(1, 2, 0)                             # (N, 4, T+1)

    n_pad = int(round(DOMAIN_PAD * (N_STEPS + 1)))             # e.g. 20 for T=131
    fc = FCLegendre(d=FC_DEGREE, n_additional_pts=n_pad)
    x_data = fc.extend(x_data, dim=1)                          # (N, 5, T+n_pad)
    y_data = fc.extend(y_data, dim=1)                          # (N, 4, T+n_pad)
    return x_data, y_data, t, fc


def rel_l2_loss(pred, target, eps=1e-8) -> torch.Tensor:
    """Per-sample relative L2, averaged over batch."""
    return ((pred - target).norm(dim=(1, 2)) / (target.norm(dim=(1, 2)) + eps)).mean()


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
        self.lift  = nn.Conv1d(IN_CH + 1, HIDDEN, 1)           # +1 for time grid channel
        self.convs = nn.ModuleList(SpectralConv1d(HIDDEN, HIDDEN, N_MODES) for _ in range(N_LAYERS))
        self.skips = nn.ModuleList(nn.Conv1d(HIDDEN, HIDDEN, 1) for _ in range(N_LAYERS))
        self.proj  = nn.Sequential(nn.Conv1d(HIDDEN, HIDDEN, 1), nn.GELU(), nn.Conv1d(HIDDEN, STATE_DIM, 1))

    def forward(self, x):  # (B, IN_CH, T_ext)
        B, _, T = x.shape
        grid = torch.linspace(0, 1, T, device=x.device).view(1, 1, T).expand(B, 1, T)
        x = self.lift(torch.cat([x, grid], dim=1))
        for i, (conv, skip) in enumerate(zip(self.convs, self.skips)):
            x = conv(x) + skip(x)
            if i < N_LAYERS - 1: x = F.gelu(x)
        return self.proj(x)


def make_model_scratch():
    return FNO1d()


# ── library FNO ───────────────────────────────────────────────────────────────
def make_model_library():
    from neuralop.models import FNO
    return FNO(
        n_modes=(N_MODES,), in_channels=IN_CH, out_channels=STATE_DIM,
        hidden_channels=HIDDEN, n_layers=N_LAYERS, positional_embedding="grid",
    )


# ── training ──────────────────────────────────────────────────────────────────
def train(x_data, y_data, fc):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  Model: {'library' if USE_LIBRARY else 'scratch'}")

    fc.to(device)
    model = (make_model_library() if USE_LIBRARY else make_model_scratch()).to(device)
    x_data, y_data = x_data.to(device), y_data.to(device)
    n = x_data.shape[0]
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS)

    losses = []
    for epoch in (bar := tqdm(range(1, N_EPOCHS+1), bar_format=BAR_FMT)):
        idx  = torch.randperm(n, device=device)[:BATCH_SIZE]
        loss = rel_l2_loss(model(x_data[idx]), y_data[idx])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        losses.append(loss.item())
        if epoch % 10 == 0: bar.set_description(f"Epoch {epoch:4d} | RelL2 {torch.tensor(losses[-10:]).mean():.2e}")
    return model, losses


# ── plotting ──────────────────────────────────────────────────────────────────
def plot_loss(losses):
    plt.figure(figsize=(7, 3))
    plt.semilogy(losses); plt.xlabel("Epoch"); plt.ylabel("Relative L2"); plt.title("Training loss")
    plt.tight_layout(); plt.savefig(LOSS_PATH, dpi=150, bbox_inches="tight"); print(f"Saved -> {LOSS_PATH}")
    if SHOW_PLOTS: plt.show()


def plot_eval(model, x_data, y_data, t, fc, n_cols=3):
    model.eval()
    labels = ["x [m]", "xd [m/s]", "θ [rad]", "θd [rad/s]"]
    device = next(model.parameters()).device

    plt.figure(figsize=(5*n_cols, 10))
    for col in range(n_cols):
        xb = x_data[col:col+1].to(device)
        with torch.no_grad(): pred = fc.restrict(model(xb), dim=1).squeeze(0).cpu()
        true = fc.restrict(y_data[col:col+1], dim=1).squeeze(0)
        for row in range(STATE_DIM):
            plt.subplot(STATE_DIM, n_cols, row*n_cols + col + 1)
            plt.plot(t, true[row], lw=2, label="True")
            plt.plot(t, pred[row], "--", lw=2, label="FNO")
            if col == 0: plt.ylabel(labels[row])
            if row == 0: plt.title(f"Traj {col+1}"); plt.legend(fontsize=8)
            if row == STATE_DIM-1: plt.xlabel("Time [s]")

    tag = "library" if USE_LIBRARY else "scratch"
    plt.suptitle(f"FNO CartPole ({tag}) — True vs Predicted", fontsize=13)
    plt.tight_layout(); plt.savefig(EVAL_PATH, dpi=150, bbox_inches="tight"); print(f"Saved -> {EVAL_PATH}")
    if SHOW_PLOTS: plt.show()


def plot_dataset(x_data, y_data, t, fc, n_cols=3):
    """Show a few raw trajectories with the FC-Legendre padding region shaded."""
    labels = ["x [m]", "xd [m/s]", "θ [rad]", "θd [rad/s]"]
    T, T_ext = len(t), x_data.shape[-1]
    n_pad = T_ext - T
    half  = n_pad // 2
    dt    = (t[-1] - t[0]) / (T - 1)
    # time axis covering the full extended domain
    t_ext = torch.cat([t[0] + dt * torch.arange(-half, 0), t, t[-1] + dt * torch.arange(1, half + 1)])

    plt.figure(figsize=(5*n_cols, 10))
    for col in range(n_cols):
        true_ext = y_data[col].cpu()                            # (4, T_ext)
        for row in range(STATE_DIM):
            plt.subplot(STATE_DIM, n_cols, row*n_cols + col + 1)
            plt.plot(t_ext, true_ext[row], lw=2)
            plt.axvspan(t_ext[0],  t[0],  alpha=0.15, label="FC pad")   # left bridge
            plt.axvspan(t[-1], t_ext[-1], alpha=0.15)                    # right bridge
            plt.axvline(t[0].item(),  lw=1, ls="--")
            plt.axvline(t[-1].item(), lw=1, ls="--")
            if col == 0: plt.ylabel(labels[row])
            if row == 0: plt.title(f"Traj {col+1}"); plt.legend(fontsize=8)
            if row == STATE_DIM-1: plt.xlabel("Time [s]")

    plt.suptitle(f"Dataset samples — FC-Legendre padding ({n_pad} pts, {half} each side)", fontsize=13)
    plt.tight_layout(); plt.savefig("imgs/fno_cartpole_dataset.png", dpi=150, bbox_inches="tight")
    print("Saved -> imgs/fno_cartpole_dataset.png")
    if SHOW_PLOTS: plt.show()


if __name__ == "__main__":
    print("Generating dataset...")
    x_data, y_data, t, fc = make_dataset()
    print(f"Dataset: {x_data.shape[0]} trajectories, {x_data.shape[-1]} time steps (T_ext={x_data.shape[-1]}, T_orig={len(t)})")
    plot_dataset(x_data.cpu(), y_data.cpu(), t, fc)
    model, losses = train(x_data, y_data, fc)
    plot_loss(losses)
    plot_eval(model, x_data.cpu(), y_data.cpu(), t, fc)
