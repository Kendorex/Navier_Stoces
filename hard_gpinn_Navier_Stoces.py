import os
import time
import warnings
warnings.filterwarnings('ignore')
os.environ.setdefault("DDE_BACKEND", "tensorflow")

import numpy as np
import matplotlib.pyplot as plt
import deepxde as dde
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['XLA_FLAGS'] = '--xla_cpu_multi_thread_eigen=false --xla_hlo_profile=false'

dde.config.set_default_float('float32')
dde.config.set_random_seed(42)
tf.config.optimizer.set_jit(False)

RHO = 1.0
NU = 0.02
U_MAX = 1.0

L = 2.0
H_IN = 0.50
H_THROAT = 0.25
X_CONTRACT_1 = 0.75
X_CONTRACT_2 = 0.90
X_EXPAND_1 = 1.10
X_EXPAND_2 = 1.25

N_DOMAIN = 1500
N_BOUNDARY = 400
N_TEST = 400

HIDDEN_LAYERS = 4
HIDDEN_WIDTH = 64
ACTIVATION = "tanh"
INITIALIZER = "Glorot uniform"

LR_ADAM = 1e-3
ITERATIONS_ADAM = 4000
DISPLAY_EVERY = 500
USE_LBFGS = False

MODEL_NAME = "gpinn_navier_stokes_constricted"
OUT_DIR = f"{MODEL_NAME}_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

GRID_NX = 420
GRID_NY = 140


points = [
    [0.0, H_IN],
    [X_CONTRACT_1, H_IN],
    [X_CONTRACT_2, H_THROAT],
    [X_EXPAND_1, H_THROAT],
    [X_EXPAND_2, H_IN],
    [L, H_IN],
    [L, -H_IN],
    [X_EXPAND_2, -H_IN],
    [X_EXPAND_1, -H_THROAT],
    [X_CONTRACT_2, -H_THROAT],
    [X_CONTRACT_1, -H_IN],
    [0.0, -H_IN],
]

geom = dde.geometry.Polygon(points)


def half_height_numpy(x):
    x = np.asarray(x)
    h = np.full_like(x, H_IN, dtype=float)
    
    mask = (x >= X_CONTRACT_1) & (x <= X_CONTRACT_2)
    h[mask] = H_IN + (H_THROAT - H_IN) * (x[mask] - X_CONTRACT_1) / (X_CONTRACT_2 - X_CONTRACT_1)
    
    mask = (x > X_CONTRACT_2) & (x < X_EXPAND_1)
    h[mask] = H_THROAT
    
    mask = (x >= X_EXPAND_1) & (x <= X_EXPAND_2)
    h[mask] = H_THROAT + (H_IN - H_THROAT) * (x[mask] - X_EXPAND_1) / (X_EXPAND_2 - X_EXPAND_1)
    
    return h


def poiseuille_profile(y, h, u_max=U_MAX):
    y = np.asarray(y)
    return u_max * (1.0 - (y / h) ** 2)


def navier_stokes_with_gradients(x, y):
    u = tf.cast(y[:, 0:1], tf.float32)
    v = tf.cast(y[:, 1:2], tf.float32)
    p = tf.cast(y[:, 2:3], tf.float32)
    
    u_x = tf.cast(dde.grad.jacobian(u, x, i=0, j=0), tf.float32)
    u_y = tf.cast(dde.grad.jacobian(u, x, i=0, j=1), tf.float32)
    v_x = tf.cast(dde.grad.jacobian(v, x, i=0, j=0), tf.float32)
    v_y = tf.cast(dde.grad.jacobian(v, x, i=0, j=1), tf.float32)
    p_x = tf.cast(dde.grad.jacobian(p, x, i=0, j=0), tf.float32)
    p_y = tf.cast(dde.grad.jacobian(p, x, i=0, j=1), tf.float32)
    
    u_xx = tf.cast(dde.grad.hessian(u, x, i=0, j=0), tf.float32)
    u_yy = tf.cast(dde.grad.hessian(u, x, i=0, j=1), tf.float32)
    v_xx = tf.cast(dde.grad.hessian(v, x, i=0, j=0), tf.float32)
    v_yy = tf.cast(dde.grad.hessian(v, x, i=0, j=1), tf.float32)
    
    RHO_f32 = tf.constant(RHO, dtype=tf.float32)
    NU_f32 = tf.constant(NU, dtype=tf.float32)
    
    momentum_x = u * u_x + v * u_y + p_x / RHO_f32 - NU_f32 * (u_xx + u_yy)
    momentum_y = u * v_x + v * v_y + p_y / RHO_f32 - NU_f32 * (v_xx + v_yy)
    continuity = u_x + v_y
    
    g_mom_x_x = tf.cast(dde.grad.jacobian(momentum_x, x, i=0, j=0), tf.float32)
    g_mom_x_y = tf.cast(dde.grad.jacobian(momentum_x, x, i=0, j=1), tf.float32)
    g_mom_y_x = tf.cast(dde.grad.jacobian(momentum_y, x, i=0, j=0), tf.float32)
    g_mom_y_y = tf.cast(dde.grad.jacobian(momentum_y, x, i=0, j=1), tf.float32)
    g_cont_x = tf.cast(dde.grad.jacobian(continuity, x, i=0, j=0), tf.float32)
    g_cont_y = tf.cast(dde.grad.jacobian(continuity, x, i=0, j=1), tf.float32)
    
    return [momentum_x, momentum_y, continuity,
            g_mom_x_x, g_mom_x_y, g_mom_y_x, g_mom_y_y, g_cont_x, g_cont_y]


def inlet(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0.0)


def outlet(x, on_boundary):
    return on_boundary and np.isclose(x[0], L)


def wall(x, on_boundary):
    return on_boundary and (not inlet(x, on_boundary)) and (not outlet(x, on_boundary))


def inlet_u(x):
    y = x[:, 1:2]
    return U_MAX * (1.0 - (y / H_IN) ** 2)


def zero(x):
    return np.zeros((len(x), 1), dtype=np.float32)


bcs = [
    dde.icbc.DirichletBC(geom, inlet_u, inlet, component=0),
    dde.icbc.DirichletBC(geom, zero, inlet, component=1),
    dde.icbc.DirichletBC(geom, zero, outlet, component=2),
    dde.icbc.DirichletBC(geom, zero, wall, component=0),
    dde.icbc.DirichletBC(geom, zero, wall, component=1),
]


data = dde.data.PDE(
    geom,
    navier_stokes_with_gradients,
    bcs,
    num_domain=N_DOMAIN,
    num_boundary=N_BOUNDARY,
    num_test=N_TEST,
)

layer_size = [2] + [HIDDEN_WIDTH] * HIDDEN_LAYERS + [3]
net = dde.nn.FNN(layer_size, ACTIVATION, INITIALIZER)
model = dde.Model(data, net)

loss_weights = [
    1.0, 1.0, 15.0,
    0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
    20.0, 20.0, 5.0, 30.0, 30.0,
]

def train_model():
    print("=" * 60)
    print("gPINN: труба с линейным сужением в середине")
    print("=" * 60)
    print(f"NU={NU}, U_MAX={U_MAX}, H_IN={H_IN}, H_THROAT={H_THROAT}, L={L}")
    print(f"Точки: внутри={N_DOMAIN}, граница={N_BOUNDARY}, тест={N_TEST}")
    print(f"Сеть: {layer_size}")
    print()

    model.compile("adam", lr=LR_ADAM, loss_weights=loss_weights)
    t0 = time.time()

    losshistory, train_state = model.train(
        iterations=ITERATIONS_ADAM,
        display_every=DISPLAY_EVERY,
    )

    print(f"Время обучения: {time.time() - t0:.2f} сек")

    if USE_LBFGS:
        print("Запуск L-BFGS...")
        model.compile("L-BFGS", loss_weights=loss_weights)
        losshistory, train_state = model.train()

    return losshistory, train_state


def predict_on_grid(nx=GRID_NX, ny=GRID_NY):
    xs = np.linspace(0.0, L, nx)
    ys = np.linspace(-H_IN, H_IN, ny)
    xx, yy = np.meshgrid(xs, ys)
    pts = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)

    inside = geom.inside(pts)
    pts_inside = pts[inside]
    pred = model.predict(pts_inside)

    u = np.full(xx.size, np.nan)
    v = np.full(xx.size, np.nan)
    p = np.full(xx.size, np.nan)
    speed = np.full(xx.size, np.nan)

    u[inside] = pred[:, 0]
    v[inside] = pred[:, 1]
    p[inside] = pred[:, 2]
    speed[inside] = np.sqrt(pred[:, 0] ** 2 + pred[:, 1] ** 2)

    return xx, yy, u.reshape(xx.shape), v.reshape(xx.shape), p.reshape(xx.shape), speed.reshape(xx.shape)


def plot_training_loss(losshistory, filename=None):
    if filename is None:
        filename = os.path.join(OUT_DIR, "training_loss.png")
    
    train_loss_total = np.sum(np.array(losshistory.loss_train), axis=1)
    test_loss_total = np.sum(np.array(losshistory.loss_test), axis=1)
    steps = np.array(losshistory.steps)
    
    plt.figure(figsize=(8, 4.2))
    plt.semilogy(steps, train_loss_total, 'b-', label='Train loss', linewidth=2)
    plt.semilogy(steps, test_loss_total, 'r--', label='Test loss', linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Total Loss")
    plt.title("gPINN Training Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=160)
    plt.close()
    print(f"Сохранено: {filename}")


def plot_results_comprehensive(xx, yy, u, v, p, speed, filename=None):
    """Комплексная визуализация в формате 2x4"""
    if filename is None:
        filename = os.path.join(OUT_DIR, "comprehensive_results.png")
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    
    # Стены канала
    x_wall = np.linspace(0, L, 500)
    y_top = half_height_numpy(x_wall)
    y_bottom = -half_height_numpy(x_wall)
    
    # u-скорость
    im1 = axes[0, 0].contourf(xx, yy, u, levels=30, cmap='RdBu_r')
    axes[0, 0].plot(x_wall, y_top, 'k-', linewidth=1.5)
    axes[0, 0].plot(x_wall, y_bottom, 'k-', linewidth=1.5)
    axes[0, 0].set_title('u-velocity', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    axes[0, 0].set_aspect('equal')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Speed
    im2 = axes[0, 1].contourf(xx, yy, speed, levels=30, cmap='plasma')
    axes[0, 1].plot(x_wall, y_top, 'k-', linewidth=1.5)
    axes[0, 1].plot(x_wall, y_bottom, 'k-', linewidth=1.5)
    axes[0, 1].set_title('Speed', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    axes[0, 1].set_aspect('equal')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Давление
    im3 = axes[0, 2].contourf(xx, yy, p, levels=30, cmap='viridis')
    axes[0, 2].plot(x_wall, y_top, 'k-', linewidth=1.5)
    axes[0, 2].plot(x_wall, y_bottom, 'k-', linewidth=1.5)
    axes[0, 2].set_title('Pressure', fontsize=12, fontweight='bold')
    axes[0, 2].set_xlabel('x')
    axes[0, 2].set_ylabel('y')
    axes[0, 2].set_aspect('equal')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # v-скорость
    im4 = axes[0, 3].contourf(xx, yy, v, levels=30, cmap='RdBu_r')
    axes[0, 3].plot(x_wall, y_top, 'k-', linewidth=1.5)
    axes[0, 3].plot(x_wall, y_bottom, 'k-', linewidth=1.5)
    axes[0, 3].set_title('v-velocity', fontsize=12, fontweight='bold')
    axes[0, 3].set_xlabel('x')
    axes[0, 3].set_ylabel('y')
    axes[0, 3].set_aspect('equal')
    plt.colorbar(im4, ax=axes[0, 3])
    
    # Профили скорости
    xs = np.linspace(0.0, L, xx.shape[1])
    ys = np.linspace(-H_IN, H_IN, xx.shape[0])
    
    throat_idx = np.argmin(np.abs(xs - 1.0))
    inlet_idx = np.argmin(np.abs(xs - 0.2))
    
    valid_inlet = ~np.isnan(u[:, inlet_idx])
    if valid_inlet.any():
        y_inlet = ys[valid_inlet]
        h_inlet = half_height_numpy(np.array([xs[inlet_idx]]))[0]
        
        axes[1, 0].plot(u[valid_inlet, inlet_idx], y_inlet, 'b-', label='gPINN', linewidth=2.5)
        u_theory = poiseuille_profile(y_inlet, h_inlet, U_MAX)
        axes[1, 0].plot(u_theory, y_inlet, 'r--', label='Poiseuille', linewidth=2)
        
        axes[1, 0].set_title('Velocity Profile at Inlet (x=0.2)', fontsize=11, fontweight='bold')
        axes[1, 0].set_xlabel('u-velocity', fontsize=10)
        axes[1, 0].set_ylabel('y', fontsize=10)
        axes[1, 0].legend(fontsize=9)
        axes[1, 0].grid(True, alpha=0.3)
    
    valid_throat = ~np.isnan(u[:, throat_idx])
    if valid_throat.any():
        y_throat = ys[valid_throat]
        h_throat = half_height_numpy(np.array([xs[throat_idx]]))[0]
        
        axes[1, 1].plot(u[valid_throat, throat_idx], y_throat, 'b-', label='gPINN', linewidth=2.5)
        u_theory = poiseuille_profile(y_throat, h_throat, U_MAX)
        axes[1, 1].plot(u_theory, y_throat, 'r--', label='Poiseuille', linewidth=2)
        
        axes[1, 1].set_title('Velocity Profile at Throat (x=1.0)', fontsize=11, fontweight='bold')
        axes[1, 1].set_xlabel('u-velocity', fontsize=10)
        axes[1, 1].set_ylabel('y', fontsize=10)
        axes[1, 1].legend(fontsize=9)
        axes[1, 1].grid(True, alpha=0.3)
    
    # Поле скорости
    step = 15
    mask = ~np.isnan(u[::step, ::step])
    X_sub = xx[::step, ::step][mask]
    Y_sub = yy[::step, ::step][mask]
    u_sub = u[::step, ::step][mask]
    v_sub = v[::step, ::step][mask]
    
    axes[1, 2].plot(x_wall, y_top, 'k-', linewidth=1.5)
    axes[1, 2].plot(x_wall, y_bottom, 'k-', linewidth=1.5)
    if len(X_sub) > 0:
        axes[1, 2].quiver(X_sub, Y_sub, u_sub, v_sub, scale=25, width=0.003)
    axes[1, 2].set_title('Velocity Field', fontsize=12, fontweight='bold')
    axes[1, 2].set_xlabel('x', fontsize=11)
    axes[1, 2].set_ylabel('y', fontsize=11)
    axes[1, 2].set_aspect('equal')
    axes[1, 2].grid(True, alpha=0.3)
    
    # Давление по центру
    center_idx = np.argmin(np.abs(ys))
    valid_center = ~np.isnan(p[center_idx, :])
    if valid_center.any():
        x_center = xs[valid_center]
        p_center = p[center_idx, valid_center]
        axes[1, 3].plot(x_center, p_center, 'b-', linewidth=2.5)
        axes[1, 3].set_title('Pressure along Centerline', fontsize=12, fontweight='bold')
        axes[1, 3].set_xlabel('x', fontsize=11)
        axes[1, 3].set_ylabel('Pressure', fontsize=11)
        axes[1, 3].grid(True, alpha=0.3)
    
    plt.suptitle('gPINN: Flow in Constricted Channel', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Сохранено: {filename}")


def create_all_outputs(losshistory):
    xx, yy, u, v, p, speed = predict_on_grid()
    
    plot_training_loss(losshistory)
    plot_results_comprehensive(xx, yy, u, v, p, speed)
    
    print(f"Все результаты сохранены в папку: {OUT_DIR}")


if __name__ == "__main__":
    losshistory, train_state = train_model()
    create_all_outputs(losshistory)
    print()
    print(f"gPINN обучение завершено! Результаты в: {OUT_DIR}")