import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.lines import Line2D

# ============================================================
# 0. REPRODUCIBILITY
# ============================================================

SEED = 0
np.random.seed(SEED)
random.seed(SEED)

plt.rcParams["figure.dpi"] = 300

# ============================================================
# 1. HYPERPARAMETERS
# ============================================================

alpha = 25.0
lr = 0.02
gamma = 0.15
lambda_irm = 5.0
lambda_eirm = 1.5
steps = 400

# Adam parameters
beta1 = 0.9
beta2 = 0.999
eps = 1e-8

env_shifts = [+0.2, -0.2]

LOSS_PLOT_MAX = 150.0
XY_PLOT_MAX   = 3.0

# ============================================================
# 2. NUMERICALLY STABLE SOFT-MIN
# ============================================================

def softmin(a, b):
    m = np.maximum(a, b)
    return m + np.log(np.exp(a - m) + np.exp(b - m))

def softmin_grad(a, b, da, db):
    m = np.maximum(a, b)
    ea = np.exp(a - m)
    eb = np.exp(b - m)
    Z = ea + eb
    return (ea * da + eb * db) / Z

# ============================================================
# 3. BASE LOSS GEOMETRY
# ============================================================

def base_loss(x, y):
    L_flat  = (x + 1)**2 + 0.1 * y**2
    L_sharp = (x - 1)**2 + alpha * y**2
    return softmin(L_flat, L_sharp)

def base_grad(x, y):
    L_flat  = (x + 1)**2 + 0.1 * y**2
    L_sharp = (x - 1)**2 + alpha * y**2

    dL_flat  = np.array([2*(x + 1), 0.2*y])
    dL_sharp = np.array([2*(x - 1), 2*alpha*y])

    return softmin_grad(L_flat, L_sharp, dL_flat, dL_sharp)

# ============================================================
# 4. ENVIRONMENT LOSSES
# ============================================================

def loss_env(theta, w, env):
    shift = env_shifts[env]
    return (w * base_loss(theta[0], theta[1]) + shift)**2

def grad_theta_env(theta, w, env):
    shift = env_shifts[env]
    L = base_loss(theta[0], theta[1])
    gL = base_grad(theta[0], theta[1])
    return 2 * (w * L + shift) * w * gL # gradient of loss_env w.r.t. theta

def grad_w_env(theta, w, env):
    shift = env_shifts[env]
    L = base_loss(theta[0], theta[1])
    return 2 * (w * L + shift) * L # gradient of loss_env w.r.t a dumpy classifier w

# ============================================================
# 5. ADAM OPTIMIZER
# ============================================================

def init_adam():
    return np.zeros(2), np.zeros(2), 0

def adam_step(theta, g, state):
    m, v, t = state
    t += 1

    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * (g ** 2)

    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)

    theta = theta - lr * m_hat / (np.sqrt(v_hat) + eps)
    return theta, (m, v, t)

# ============================================================
# 6. OPTIMIZATION STEPS
# ============================================================

def ERM_step(theta, state):
    g = sum(grad_theta_env(theta, 1.0, e) for e in range(2)) / 2
    return adam_step(theta, g, state)

def IRM_step(theta, state):
    w = 1.0
    g_theta = np.zeros(2)
    penalty = 0.0

    for e in range(2):
        g_theta += grad_theta_env(theta, w, e)
        penalty += grad_w_env(theta, w, e)**2

    g_theta /= 2
    penalty /= 2

    g = g_theta + lambda_irm * penalty * g_theta # gradient of the IRM objective function w.r.t. theta
    return adam_step(theta, g, state)

def EIRM_step(theta, state):
    g1 = sum(grad_theta_env(theta, 1.0, e) for e in range(2)) / 2
    theta_pert = theta + gamma * g1

    g2 = sum(grad_theta_env(theta_pert, 1.0, e) for e in range(2)) / 2

    L1 = sum(loss_env(theta, 1.0, e) for e in range(2)) / 2
    L2 = sum(loss_env(theta_pert, 1.0, e) for e in range(2)) / 2

    g = g1 + lambda_eirm * (L2 - L1) * (g2 - g1)
    return adam_step(theta, g, state)

# ============================================================
# 7. THREE INITIALIZATION POINTS
# ============================================================

inits = [
    np.array([0.7,  0.6]),
    np.array([1.2,  0.2]),
    np.array([0.9, -0.5]),
]

def run_traj(theta0, step_fn):
    theta = theta0.copy()
    state = init_adam()
    traj = [theta.copy()]
    for _ in range(steps):
        theta, state = step_fn(theta, state)
        traj.append(theta.copy())
    return np.array(traj)

traj_erm  = [run_traj(t0, ERM_step)  for t0 in inits]
traj_irm  = [run_traj(t0, IRM_step)  for t0 in inits]
traj_eirm = [run_traj(t0, EIRM_step) for t0 in inits]

# ============================================================
# 8. LOSS SURFACE
# ============================================================

xs = np.linspace(-2, 2, 200)
ys = np.linspace(-2, 2, 200)
X, Y = np.meshgrid(xs, ys)
Z = np.clip(base_loss(X, Y), None, LOSS_PLOT_MAX)
zmin, zmax = Z.min(), Z.max()

def filter_traj(traj):
    xs, ys = traj[:, 0], traj[:, 1]
    zs = base_loss(xs, ys)
    mask = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(zs)
    return xs[mask], ys[mask], zs[mask]

# ============================================================
# 9. LEGEND HANDLES (INIT / FINAL)
# ============================================================

legend_elements = [
    Line2D(
        [0], [0],
        marker="*",
        color="w",
        markerfacecolor="limegreen",
        markeredgecolor="k",
        markersize=14,
        label="Init"
    ),
    Line2D(
        [0], [0],
        marker="X",
        color="w",
        markerfacecolor="black",
        markersize=10,
        label="Final"
    ),
]

# ============================================================
# 10. PLOTTING
# ============================================================

fig = plt.figure(figsize=(18, 6))
axes = [
    fig.add_subplot(1, 3, 1, projection="3d"),
    fig.add_subplot(1, 3, 2, projection="3d"),
    fig.add_subplot(1, 3, 3, projection="3d"),
]

configs = [
    ("ERM",  traj_erm,  "tab:blue"),
    ("IRM",  traj_irm,  "tab:orange"),
    ("EIRM", traj_eirm, "tab:green"),
]

linestyles = ["solid", "dashed", "dotted"]

for ax, (name, trajs, color) in zip(axes, configs):

    ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.45, linewidth=0)

    for traj, ls in zip(trajs, linestyles):
        tx, ty, tz = filter_traj(traj)

        ax.plot(
            tx, ty,
            tz + 0.02 * (zmax - zmin),
            color=color,
            linewidth=2,
            linestyle=ls
        )

        # init
        ax.scatter(
            traj[0, 0], traj[0, 1], tz[0],
            c="limegreen", s=110, marker="*", edgecolor="k"
        )

        # final
        ax.scatter(
            tx[-1], ty[-1], tz[-1],
            c="black", s=80, marker="X"
        )

    # basin centers
    ax.scatter(
        [-1, 1], [0, 0],
        [base_loss(-1, 0), base_loss(1, 0)],
        c="red", s=60
    )

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(zmin, zmax)
    ax.view_init(elev=35, azim=-60)
    ax.set_box_aspect((1, 1, 0.5))
    ax.set_title(name)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Loss")

    # add legend once
    if name == "ERM":
        ax.legend(
            handles=legend_elements,
            loc="upper left",
            frameon=True
        )

plt.suptitle(
    "ERM vs IRM vs EIRM — Three Initializations per Method",
    fontsize=14
)

plt.tight_layout()
plt.savefig("three_inits_trajectories.jpg", dpi=300, bbox_inches="tight")
plt.show()