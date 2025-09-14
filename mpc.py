import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import cvxpy as cp
import time
import warnings

# =========================
# System parameters
# =========================
rho, Af, cd = 1.2, 2.2, 0.3
v_bar, meq = 20.0, 1500.0

# States: x = [v_h, d]; Inputs: u = [F_t, F_b]; Disturbance/preview: v_p
a11 = -(rho * Af * cd * v_bar) / meq
A_cont = np.array([[a11, 0.0],
                   [-1.0, 0.0]])
B_cont = np.array([[1/meq, -1/meq],
                   [0.0,    0.0]])
E_cont = np.array([[0.0],
                   [1.0]])

# =========================
# Linearization using ZOH
# =========================
Ts = 0.5
B_aug = np.hstack([B_cont, E_cont])  # [F_t, F_b, v_p]
Ad, B_aug_d, _, _, _ = signal.cont2discrete(
    (A_cont, B_aug, np.eye(2), np.zeros((2, 3))), Ts, method="zoh"
)
Bd = B_aug_d[:, :2]
Ed = B_aug_d[:, 2:3]

# =========================
# MPC configuration (fast, safe, robust) modes
# =========================
H = 15

# Tracking: emphasize distance strongly; strong terminal pull to d_ref
Q = np.diag([60.0, 800.0])          # [speed, distance]
P = np.diag([60.0, 6000.0])         # big terminal weight on distance

# Effort proxy and comfort
R = np.diag([0.06, 0.25])           # traction cost ↑; brake heavier
LAMBDA_DU = 2e-4                    # Δu smoothing
EPS_REG = 1e-8                      # tiny regularizer for strict convexity

# Safety (soft each stage)
RHO_SLACK = 1e7                     # very strong penalty on slack

# Numerics (keep states in sane boxes)
V_MIN, V_MAX = 0.0, 55.0
D_MIN, D_MAX = 0.0, 200.0

# =========================
# Scenario
# =========================
N_sim = 180
v_ref_nominal = 20.0                # shown only as a reference line
d_ref = 10.0
d_safe = 3.0
vp_profile = np.ones(N_sim + H) * 20.0  # constant lead @ 20 m/s

# Initial state
x_sim = np.zeros((2, N_sim + 1))
x_sim[:, 0] = [5.0, 15.0]           # start slower than lead, 15 m back

Ft_all, Fb_all = [], []
solve_time, solve_status = [], []

u_prev = np.zeros(2)                # for Δu at k=0

# Closing-rule parameters (anchored to lead)
K_CLOSE = 0.10                      # proportional on (d - d_ref) [m/s per m]
DEADBAND = 0.5                      # ignore small gap error [m]
REL_CAP = 1.5                       # |v_ref - v_p| ≤ 1.5 m/s (tight ⇒ no pumping)

def deadband(e, db):
    if abs(e) <= db:
        return 0.0
    return e - np.sign(e) * db

# =========================
# Closed-loop MPC
# =========================
for t in range(N_sim):
    vp_now = float(vp_profile[t])
    d_err = float(x_sim[1, t] - d_ref)

    # Effective reference speed anchored to lead with deadband and tight cap
    v_bias = K_CLOSE * deadband(d_err, DEADBAND)
    v_ref_eff = np.clip(vp_now + v_bias, vp_now - REL_CAP, vp_now + REL_CAP)
    xref = np.array([v_ref_eff, d_ref])

    # Decision variables
    x  = cp.Variable((2, H + 1))
    u  = cp.Variable((2, H))
    du = cp.Variable((2, H))
    eps = cp.Variable(H + 1, nonneg=True)

    cost = 0
    constr = [x[:, 0] == x_sim[:, t]]
    vp_h = vp_profile[t:t + H]

    for k in range(H):
        dx = x[:, k] - xref
        cost += cp.quad_form(dx, Q) + cp.quad_form(u[:, k], R)
        cost += EPS_REG * (cp.sum_squares(x[:, k]) + cp.sum_squares(u[:, k]))

        # Dynamics with vp as affine term
        constr += [x[:, k + 1] == Ad @ x[:, k] + Bd @ u[:, k] + Ed.flatten() * vp_h[k]]

        # Inputs
        constr += [0 <= u[0, k], u[0, k] <= 4000]   # traction can't go beyond 4000 N
        constr += [0 <= u[1, k], u[1, k] <= 4000]   # braking can't go beyond 4000 N

        # State boxes
        constr += [V_MIN <= x[0, k], x[0, k] <= V_MAX,
                   D_MIN <= x[1, k], x[1, k] <= D_MAX]

        # Δu smoothing
        if k == 0:
            constr += [du[:, k] == u[:, k] - u_prev]
        else:
            constr += [du[:, k] == u[:, k] - u[:, k-1]]
        cost += LAMBDA_DU * cp.sum_squares(du[:, k])

        # Stage soft-safety
        constr += [x[1, k] + eps[k] >= d_safe]

    # Terminal cost and constraints
    cost += cp.quad_form(x[:, H] - xref, P) + EPS_REG * cp.sum_squares(x[:, H])
    constr += [V_MIN <= x[0, H], x[0, H] <= V_MAX,
               D_MIN <= x[1, H], x[1, H] <= D_MAX,
               x[1, H] + eps[H] >= d_safe]
    cost += RHO_SLACK * cp.sum(eps)

    # Solve fast (OSQP), fallback SCS if needed
    prob = cp.Problem(cp.Minimize(cost), constr)
    t0 = time.time()
    prob.solve(solver=cp.OSQP, verbose=False, eps_abs=5e-4, eps_rel=5e-4,
               max_iter=60000, polish=True, warm_start=True, adaptive_rho=True)
    status = prob.status
    if status not in ["optimal", "optimal_inaccurate"]:
        prob.solve(solver=cp.SCS, verbose=False, eps=1e-4, max_iters=120000,
                   acceleration_lookback=30, alpha=1.8)
        status = prob.status

    solve_time.append(time.time() - t0)
    solve_status.append(status)
    if status not in ["optimal", "optimal_inaccurate"]:
        warnings.warn(f"Step {t}: solver {status}. Holding previous u.")
        u0 = u_prev.copy()
    else:
        u0 = u[:, 0].value

    Ft_all.append(float(u0[0]))
    Fb_all.append(float(u0[1]))

    # Plant update and warm start
    x_sim[:, t + 1] = Ad @ x_sim[:, t] + Bd @ u0 + (Ed.flatten() * vp_now)
    u_prev = u0.copy()

# =========================
# Metrics (use SAME v_ref definition)
# =========================
Ft_all = np.array(Ft_all)
Fb_all = np.array(Fb_all)
vh = x_sim[0, :len(Ft_all)]
d  = x_sim[1, :len(Ft_all)]
t_axis = np.arange(len(Ft_all)) * Ts

vh_ref_series = np.zeros_like(vh)
for k in range(len(vh)):
    vp_now = vp_profile[k]
    d_err  = d[k] - d_ref
    v_bias = K_CLOSE * deadband(d_err, DEADBAND)
    vh_ref_series[k] = np.clip(vp_now + v_bias, vp_now - REL_CAP, vp_now + REL_CAP)

rmse_v = np.sqrt(np.mean((vh - vh_ref_series) ** 2))
gap_margin_min = np.min(d - d_safe)
energy_proxy = np.sum(Ft_all**2 + Fb_all**2)
mean_solve = np.mean(solve_time)
p95_solve  = np.percentile(solve_time, 95)

print(f"Solved steps: {len(Ft_all)} / {N_sim}. Last status: {solve_status[-1]}")
print(f"Final vh = {vh[-1]:.3f} m/s, Final d = {d[-1]:.3f} m")
print(f"RMSE(v) = {rmse_v:.3f}, min(d - d_safe) = {gap_margin_min:.3f} m")
print(f"Energy proxy ∑u^2 = {energy_proxy:.1f}")
print(f"Solve time: mean {1e3*mean_solve:.2f} ms, p95 {1e3*p95_solve:.2f} ms")

# =========================
# Plots
# =========================
fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

axs[0].plot(t_axis, vh, label='Host Velocity (vh)')
axs[0].plot(t_axis, vp_profile[:len(t_axis)], '--', label='Lead Velocity (vp)')
axs[0].plot(t_axis, vh_ref_series, ':', label='Effective v_ref')
axs[0].axhline(v_ref_nominal, ls='--', label='Nominal v_ref')
axs[0].set_ylabel('Velocity [m/s]')
axs[0].legend()
axs[0].grid(True)

axs[1].plot(t_axis, d, label='Gap d')
axs[1].axhline(d_safe, ls='--', label='Safe Distance')
axs[1].axhline(d_ref, ls=':', label='Desired Distance')
axs[1].set_ylabel('Distance [m]')
axs[1].legend()
axs[1].grid(True)

axs[2].plot(t_axis, Ft_all, label='Traction Ft')
axs[2].plot(t_axis, Fb_all, label='Braking Fb')
axs[2].set_xlabel('Time [s]')
axs[2].set_ylabel('Force [N]')
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.savefig('mpc_analysis.png', dpi=300, bbox_inches='tight')
plt.show()