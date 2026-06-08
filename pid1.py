import numpy as np
from tqdm import tqdm

import cart


N = 1
DT = 1e-3
T = 8.0
T_CTRL = 0.02
U_MAX = 80.0
X_WALL = 6.0
KP = 500.0
KI = 0.0
KD = 80.0
KX = 0.05
KDX = 0.3


def wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class PID:
    def __init__(self, kp=KP, ki=KI, kd=KD, u_max=U_MAX):
        self.kp, self.ki, self.kd, self.u_max = kp, ki, kd, u_max
        self.i = 0.0; self.prev_e = 0.0

    def __call__(self, e, de, dt):
        self.i += e * dt
        u = self.kp * e + self.ki * self.i + self.kd * de
        u_clip = float(np.clip(u, -self.u_max, self.u_max))
        if u != u_clip: self.i -= e * dt  # anti-windup for saturation
        return u_clip


def policy(z, pid, dt_ctrl=T_CTRL):
    x, th, dx, dth = z
    e = wrap_pi(th - np.pi)
    u = -pid(e, dth, dt_ctrl) - KX * x - KDX * dx
    return float(np.clip(u, -U_MAX, U_MAX))


def simulate_pid(z0, t=T, dt=DT, t_ctrl=T_CTRL, show_progress=True):
    ts = np.arange(0, t + dt, dt); z = np.zeros((len(ts), len(z0))); u = np.zeros(len(ts))
    pid = PID(); z[0] = z0; hold = policy(z0, pid, t_ctrl); next_ctrl = 0.0
    for k in tqdm(range(len(ts) - 1), disable=not show_progress):
        if ts[k] >= next_ctrl: hold = policy(z[k], pid, t_ctrl); next_ctrl += t_ctrl
        u[k] = hold; z[k + 1] = cart.rk4_step(z[k], dt=dt, u=hold, n=N, x_wall=X_WALL)
    u[-1] = u[-2]
    return ts, z, u


if __name__ == "__main__":
    z0 = np.array([0.0, np.pi + 0.03, 0.0, 0.0])
    ts, traj, u = simulate_pid(z0, t=T, dt=DT, t_ctrl=T_CTRL)
    cart.animate_trajectory(traj, dt=DT, n=N, u=u, u_scale=0.02, x_wall=X_WALL, stride=20)
