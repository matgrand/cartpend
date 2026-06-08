import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm


G = 9.81
M = 1.0
LINK_MASS = 0.2
LINK_LENGTH = 1.0
N_ARMS = 1
DT = 1e-3
T = 10.0
SHOW_PLOTS = True
WALL_X = 6.0
WALL_RESTITUTION = 0.0


def _vec(x, n):
    x = np.asarray(x, dtype=float)
    return np.full(n, float(x)) if x.ndim == 0 else x


def arm_params(n=N_ARMS, m=LINK_MASS, l=LINK_LENGTH):
    m, l = _vec(m, n), _vec(l, n)
    if len(m) != n or len(l) != n: raise ValueError("m and l must be scalars or length n")
    return m, l, np.array([np.sum(m[i:]) for i in range(n)])


def dynamics(z, u=0.0, n=N_ARMS, cart_mass=M, m=LINK_MASS, l=LINK_LENGTH, g=G):
    """Return dz/dt for an n-link pendulum on a cart, with cart force u."""
    z = np.asarray(z, dtype=float); q, dq = z[: n + 1], z[n + 1 :]
    th, dth = q[1:], dq[1:]; m, l, sm = arm_params(n, m, l)
    a = np.zeros((n + 1, n + 1)); b = np.zeros(n + 1)

    a[0, 0] = cart_mass + np.sum(m)
    a[0, 1:] = sm * l * np.cos(th); a[1:, 0] = a[0, 1:]
    b[0] = float(u) + np.sum(sm * l * np.sin(th) * dth**2)

    for i in range(n):
        for j in range(n):
            sij = sm[max(i, j)]
            a[i + 1, j + 1] = sij * l[i] * l[j] * np.cos(th[i] - th[j])
            b[i + 1] -= sij * l[i] * l[j] * np.sin(th[i] - th[j]) * dth[j] ** 2
        b[i + 1] -= sm[i] * g * l[i] * np.sin(th[i])

    ddq = np.linalg.solve(a, b)
    return np.r_[dq, ddq]


def apply_wall(z, x_wall=WALL_X, restitution=WALL_RESTITUTION, n=N_ARMS):
    z = np.array(z, dtype=float, copy=True); dx_i = n + 1
    if z[0] < -x_wall: z[0] = -x_wall; z[dx_i] = max(-restitution * z[dx_i], 0.0)
    if z[0] > x_wall: z[0] = x_wall; z[dx_i] = min(-restitution * z[dx_i], 0.0)
    return z


def rk4_step(z, dt=DT, u=0.0, f=dynamics, x_wall=None, **kw):
    """One RK4 step. u can be a scalar force or a callable u(t, z) when t is in kw."""
    t = kw.pop("t", 0.0)
    force = u if callable(u) else lambda _t, _z: u
    k1 = f(z, force(t, z), **kw)
    k2 = f(z + dt * k1 / 2, force(t + dt / 2, z + dt * k1 / 2), **kw)
    k3 = f(z + dt * k2 / 2, force(t + dt / 2, z + dt * k2 / 2), **kw)
    k4 = f(z + dt * k3, force(t + dt, z + dt * k3), **kw)
    z_next = z + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return apply_wall(z_next, x_wall=x_wall, n=kw.get("n", N_ARMS)) if x_wall is not None else z_next


def simulate(z0, t=T, dt=DT, u=0.0, n=N_ARMS, x_wall=None, show_progress=True, **kw):
    """Simulate for t seconds. Returns times, trajectory."""
    ts = np.arange(0, t + dt, dt); z = np.zeros((len(ts), len(z0))); z[0] = z0
    it = tqdm(range(len(ts) - 1), disable=not show_progress)
    for k in it: z[k + 1] = rk4_step(z[k], dt, u, t=ts[k], n=n, x_wall=x_wall, **kw)
    return ts, z


def cartesian_points(z, n=N_ARMS, l=LINK_LENGTH):
    q = np.asarray(z)[: n + 1]; x, th = q[0], q[1:]; l = _vec(l, n)
    xs = np.r_[x, x + np.cumsum(l * np.sin(th))]
    ys = np.r_[0.0, -np.cumsum(l * np.cos(th))]
    return xs, ys


def animate_trajectory(traj, dt=DT, n=N_ARMS, l=LINK_LENGTH, u=None, u_scale=0.08, x_wall=WALL_X, stride=20, save=None, show=SHOW_PLOTS):
    """Animate a full trajectory array returned by simulate."""
    traj = np.asarray(traj); frames = np.arange(0, len(traj), stride); ts = np.arange(len(traj)) * dt
    u = np.zeros(len(traj)) if u is None else np.asarray(u, dtype=float)
    if len(u) != len(traj): raise ValueError("u must have the same length as traj")
    pts = [cartesian_points(z, n, l) for z in traj[frames]]
    lim = 1.1 * np.sum(_vec(l, n))

    fig = plt.figure(figsize=(12.8, 7.2)); ax = plt.subplot(1, 2, 1); ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-x_wall, x_wall); ax.set_ylim(-lim, lim); ax.set_xlabel("x"); ax.set_ylabel("y")
    line, = plt.plot([], [], "o-", lw=2); cart, = plt.plot([], [], "s", ms=12); ctrl, = plt.plot([], [], "r-", lw=3)

    markers = []
    plt.subplot(3, 2, 2); plt.plot(ts, traj[:, 0], label="x"); plt.plot(ts, traj[:, n + 1], label="dx")
    if u is not None: plt.plot(ts, u, "r", label="u")
    markers.append(plt.axvline(0.0, color="yellow", lw=2)); plt.ylabel("cart"); plt.legend(ncol=2)
    plt.subplot(3, 2, 4)
    for i in range(n): plt.plot(ts, traj[:, i + 1], label=f"th{i + 1}")
    markers.append(plt.axvline(0.0, color="yellow", lw=2)); plt.ylabel("angles"); plt.legend(ncol=2)
    plt.subplot(3, 2, 6)
    for i in range(n): plt.plot(ts, traj[:, n + 2 + i], label=f"dth{i + 1}")
    markers.append(plt.axvline(0.0, color="yellow", lw=2)); plt.xlabel("time [s]"); plt.ylabel("dangles"); plt.legend(ncol=2)

    def update(k):
        xs, ys = pts[k]; line.set_data(xs, ys); cart.set_data([xs[0]], [0.0])
        ctrl.set_data([xs[0], xs[0] + u_scale * u[frames[k]]], [0.0, 0.0])
        for marker in markers: marker.set_xdata([ts[frames[k]], ts[frames[k]]])
        return line, cart, ctrl, *markers

    anim = FuncAnimation(fig, update, frames=len(pts), interval=1000 * dt * stride, blit=False)
    plt.tight_layout()
    if save is not None: anim.save(save); print(f"saved animation to {save}")
    if show: plt.show()
    return anim


if __name__ == "__main__":
    n = 1
    z0 = np.zeros(2 * (n + 1)); z0[1] = 0.7; z0[n + 2] = 0.0
    ts, traj = simulate(z0, t=8.0, dt=2e-3, u=0.0, n=n)
    animate_trajectory(traj, dt=2e-3, n=n, stride=10)
