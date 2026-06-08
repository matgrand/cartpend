import numpy as np
from tqdm import tqdm

import cart


N_ARMS = 4
DT = 1e-3
T = 5.0
T_CTRL = 1e-2#0.02
TF = 1.0
N_HORIZON = 6*25
U_MAX = 2*80.0
X_MAX = 4.0
X_WALL = 6.0
MASS_MISMATCH = 0.0
MISMATCH_SEED = 1
ARM_WEIGHT_MIN = 0.25
ARM_WEIGHT_MAX = 1.0
IC_X = 0.0
IC_TH_OFFSET = 0.04 + np.pi
IC_DX = 0.0
IC_DTH = 0.0
QP_SOLVER = "PARTIAL_CONDENSING_HPIPM"
INTEGRATOR = "ERK" #"IRK"
# NLP_SOLVER = "SQP" #"SQP_RTI"
NLP_SOLVER = "SQP_RTI"


def _require_acados():
    try:
        import casadi as ca
        from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
    except ModuleNotFoundError as e:
        msg = "NMPC needs casadi and acados_template installed and ACADOS_SOURCE_DIR/LD_LIBRARY_PATH configured."
        raise ModuleNotFoundError(msg) from e
    return ca, AcadosModel, AcadosOcp, AcadosOcpSolver


def upright_state(n=N_ARMS, x=0.0):
    z = np.zeros(2 * (n + 1)); z[0] = x; z[1 : n + 1] = np.pi
    return z


def arm_priority_weights(n=N_ARMS, w_min=ARM_WEIGHT_MIN, w_max=ARM_WEIGHT_MAX):
    if n == 1: return np.array([w_max])
    return np.linspace(w_min, w_max, n)


def demo_initial_state(n=N_ARMS, x=IC_X, th_offset=IC_TH_OFFSET, dx=IC_DX, dth=IC_DTH):
    z = upright_state(n, x=x); z[1 : n + 1] += np.asarray(th_offset); z[n + 1] = dx
    z[n + 2 :] = np.asarray(dth)
    return z


def export_model(n=N_ARMS, cart_mass=cart.M, m=cart.LINK_MASS, l=cart.LINK_LENGTH, g=cart.G):
    ca, AcadosModel, _, _ = _require_acados()
    m, l, sm = cart.arm_params(n, m, l)
    x = ca.SX.sym("x", 2 * (n + 1)); xdot = ca.SX.sym("xdot", 2 * (n + 1)); u = ca.SX.sym("u", 1)
    q, dq = x[: n + 1], x[n + 1 :]; th, dth = q[1:], dq[1:]
    a = ca.SX.zeros(n + 1, n + 1); b = ca.SX.zeros(n + 1)

    a[0, 0] = cart_mass + np.sum(m)
    for i in range(n):
        a[0, i + 1] = sm[i] * l[i] * ca.cos(th[i]); a[i + 1, 0] = a[0, i + 1]
        b[0] += sm[i] * l[i] * ca.sin(th[i]) * dth[i] ** 2
    b[0] += u[0]

    for i in range(n):
        for j in range(n):
            sij = sm[max(i, j)]
            a[i + 1, j + 1] = sij * l[i] * l[j] * ca.cos(th[i] - th[j])
            b[i + 1] -= sij * l[i] * l[j] * ca.sin(th[i] - th[j]) * dth[j] ** 2
        b[i + 1] -= sm[i] * g * l[i] * ca.sin(th[i])

    f_expl = ca.vertcat(dq, ca.solve(a, b))
    model = AcadosModel(); model.name = f"cartpend_{n}arm"; model.x = x; model.xdot = xdot; model.u = u
    model.f_expl_expr = f_expl; model.f_impl_expr = xdot - f_expl
    model.cost_y_expr = ca.vertcat(x[0], x[n + 1], ca.sin(th), ca.cos(th) + 1, dth, u)
    model.cost_y_expr_e = ca.vertcat(x[0], x[n + 1], ca.sin(th), ca.cos(th) + 1, dth)
    return model


def make_ocp(n=N_ARMS, tf=TF, n_horizon=N_HORIZON, u_max=U_MAX, x_max=X_MAX):
    _, _, AcadosOcp, AcadosOcpSolver = _require_acados()
    model = export_model(n); nx = 2 * (n + 1); ny = 3 * n + 3; ny_e = 3 * n + 2
    ocp = AcadosOcp(); ocp.model = model
    ocp.solver_options.N_horizon = n_horizon; ocp.solver_options.tf = tf

    aw = arm_priority_weights(n)
    wy = np.r_[2.0, 0.2, 500.0 * aw, 500.0 * aw, 10.0 * aw, 0.02]
    wy_e = np.r_[10.0, 1.0, 1500.0 * aw, 1500.0 * aw, 50.0 * aw]
    ocp.cost.cost_type = "NONLINEAR_LS"; ocp.cost.W = np.diag(wy); ocp.cost.yref = np.zeros(ny)
    ocp.cost.cost_type_e = "NONLINEAR_LS"; ocp.cost.W_e = np.diag(wy_e); ocp.cost.yref_e = np.zeros(ny_e)

    ocp.constraints.x0 = upright_state(n)
    ocp.constraints.lbu = np.array([-u_max]); ocp.constraints.ubu = np.array([u_max]); ocp.constraints.idxbu = np.array([0])
    ocp.constraints.lbx = np.array([-x_max]); ocp.constraints.ubx = np.array([x_max]); ocp.constraints.idxbx = np.array([0])

    ocp.solver_options.qp_solver = QP_SOLVER
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = INTEGRATOR
    ocp.solver_options.nlp_solver_type = NLP_SOLVER
    ocp.solver_options.globalization = "MERIT_BACKTRACKING"
    return ocp, AcadosOcpSolver(ocp, json_file=f"acados_ocp_{model.name}.json")


class NMPC:
    def __init__(self, n=N_ARMS, tf=TF, n_horizon=N_HORIZON, u_max=U_MAX):
        self.n, self.u_max = n, u_max
        self.ocp, self.solver = make_ocp(n=n, tf=tf, n_horizon=n_horizon, u_max=u_max)

    def __call__(self, z):
        z = np.asarray(z, dtype=float)
        self.solver.set(0, "lbx", z); self.solver.set(0, "ubx", z)
        status = self.solver.solve()
        if status != 0: print(f"acados status {status}, using clipped previous/control guess")
        return float(np.clip(self.solver.get(0, "u")[0], -self.u_max, self.u_max))


def simulate_nmpc(z0, n=N_ARMS, t=T, dt=DT, t_ctrl=T_CTRL, mass_mismatch=MASS_MISMATCH, mismatch_seed=MISMATCH_SEED, show_progress=True):
    ts = np.arange(0, t + dt, dt); z = np.zeros((len(ts), len(z0))); u = np.zeros(len(ts))
    plant_cart_mass, plant_m = cart.random_masses(n, pct=mass_mismatch, seed=mismatch_seed)
    ctrl = NMPC(n=n); z[0] = z0; hold = 0.0; next_ctrl = 0.0
    for k in tqdm(range(len(ts) - 1), disable=not show_progress):
        if ts[k] >= next_ctrl: hold = ctrl(z[k]); next_ctrl += t_ctrl
        u[k] = hold; z[k + 1] = cart.rk4_step(z[k], dt=dt, u=hold, n=n, cart_mass=plant_cart_mass, m=plant_m, x_wall=X_WALL)
    u[-1] = u[-2]
    return ts, z, u


if __name__ == "__main__":
    n = N_ARMS
    z0 = demo_initial_state(n)
    ts, traj, u = simulate_nmpc(z0, n=n, t=T, dt=DT, t_ctrl=T_CTRL)
    cart.animate_trajectory(traj, dt=DT, n=n, u=u, u_scale=0.02, x_wall=X_WALL, stride=20)
