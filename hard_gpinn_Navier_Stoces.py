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

# Установка float32 как типа по умолчанию
dde.config.set_default_float('float32')
dde.config.set_random_seed(42)
tf.config.optimizer.set_jit(False)


# ============================================================
# gPINN Navier-Stokes: труба с линейным сужением в середине
# ============================================================

# -------------------------
# 1. Настройки
# -------------------------

# Физика
RHO = 1.0
NU = 0.02
U_MAX = 1.0

# Геометрия
L = 2.0
H_IN = 0.50
H_THROAT = 0.25
X_CONTRACT_1 = 0.75
X_CONTRACT_2 = 0.90
X_EXPAND_1 = 1.10
X_EXPAND_2 = 1.25

# Обучение
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

# Вывод
MODEL_NAME = "gpinn_navier_stokes_constricted"
OUT_DIR = f"{MODEL_NAME}_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

GRID_NX = 420
GRID_NY = 140


# -------------------------
# 2. Геометрия
# -------------------------

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


# -------------------------
# 3. Теоретический профиль Пуазейля для сравнения
# -------------------------

def poiseuille_profile(y, h, u_max=U_MAX):
    """
    Профиль Пуазейля для канала высотой 2h
    u(y) = u_max * (1 - (y/h)^2)
    """
    y = np.asarray(y)
    return u_max * (1.0 - (y / h) ** 2)


def get_local_poiseuille_profile(x, y):
    """
    Локальный профиль Пуазейля с учетом переменной высоты канала
    Используется только для входного и выходного сечений
    """
    h_local = half_height_numpy(x)
    return poiseuille_profile(y, h_local, U_MAX)


# -------------------------
# 4. gPINN: Уравнения Навье-Стокса + градиенты
# -------------------------

def navier_stokes_with_gradients(x, y):
    """
    gPINN: возвращает невязки PDE и их градиенты по x и y
    """
    u = tf.cast(y[:, 0:1], tf.float32)
    v = tf.cast(y[:, 1:2], tf.float32)
    p = tf.cast(y[:, 2:3], tf.float32)
    
    # Производные первого порядка
    u_x = tf.cast(dde.grad.jacobian(u, x, i=0, j=0), tf.float32)
    u_y = tf.cast(dde.grad.jacobian(u, x, i=0, j=1), tf.float32)
    v_x = tf.cast(dde.grad.jacobian(v, x, i=0, j=0), tf.float32)
    v_y = tf.cast(dde.grad.jacobian(v, x, i=0, j=1), tf.float32)
    p_x = tf.cast(dde.grad.jacobian(p, x, i=0, j=0), tf.float32)
    p_y = tf.cast(dde.grad.jacobian(p, x, i=0, j=1), tf.float32)
    
    # Производные второго порядка
    u_xx = tf.cast(dde.grad.hessian(u, x, i=0, j=0), tf.float32)
    u_yy = tf.cast(dde.grad.hessian(u, x, i=0, j=1), tf.float32)
    v_xx = tf.cast(dde.grad.hessian(v, x, i=0, j=0), tf.float32)
    v_yy = tf.cast(dde.grad.hessian(v, x, i=0, j=1), tf.float32)
    
    # Константы
    RHO_f32 = tf.constant(RHO, dtype=tf.float32)
    NU_f32 = tf.constant(NU, dtype=tf.float32)
    
    # Невязки PDE
    momentum_x = u * u_x + v * u_y + p_x / RHO_f32 - NU_f32 * (u_xx + u_yy)
    momentum_y = u * v_x + v * v_y + p_y / RHO_f32 - NU_f32 * (v_xx + v_yy)
    continuity = u_x + v_y
    
    # Градиенты невязок (gPINN)
    g_mom_x_x = tf.cast(dde.grad.jacobian(momentum_x, x, i=0, j=0), tf.float32)
    g_mom_x_y = tf.cast(dde.grad.jacobian(momentum_x, x, i=0, j=1), tf.float32)
    g_mom_y_x = tf.cast(dde.grad.jacobian(momentum_y, x, i=0, j=0), tf.float32)
    g_mom_y_y = tf.cast(dde.grad.jacobian(momentum_y, x, i=0, j=1), tf.float32)
    g_cont_x = tf.cast(dde.grad.jacobian(continuity, x, i=0, j=0), tf.float32)
    g_cont_y = tf.cast(dde.grad.jacobian(continuity, x, i=0, j=1), tf.float32)
    
    return [momentum_x, momentum_y, continuity,
            g_mom_x_x, g_mom_x_y, g_mom_y_x, g_mom_y_y, g_cont_x, g_cont_y]


# -------------------------
# 5. Граничные условия
# -------------------------

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


# -------------------------
# 6. Данные и сеть
# -------------------------

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


# -------------------------
# 7. Обучение
# -------------------------

def train_model():
    print("=" * 60)
    print("gPINN: труба с линейным сужением в середине")
    print("=" * 60)
    print(f"NU={NU}, U_MAX={U_MAX}, H_IN={H_IN}, H_THROAT={H_THROAT}, L={L}")
    print(f"Точки: внутри={N_DOMAIN}, граница={N_BOUNDARY}, тест={N_TEST}")
    print(f"Сеть: {layer_size}")
    print(f"gPINN: 3 PDE + 6 градиентов PDE в loss")
    print(f"Adam итерации: {ITERATIONS_ADAM}")
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


# -------------------------
# 8. Вспомогательные функции
# -------------------------

def _stack_operator_result(res):
    if isinstance(res, list):
        return np.hstack([np.asarray(r).reshape(-1, 1) for r in res])
    res = np.asarray(res)
    if res.ndim == 1:
        return res.reshape(-1, 1)
    return res


def predict_on_grid(nx=GRID_NX, ny=GRID_NY):
    xs = np.linspace(0.0, L, nx)
    ys = np.linspace(-H_IN, H_IN, ny)
    xx, yy = np.meshgrid(xs, ys)
    pts = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)

    inside = geom.inside(pts)
    pts_inside = pts[inside]
    pred = model.predict(pts_inside)
    res = _stack_operator_result(model.predict(pts_inside, operator=navier_stokes_with_gradients))
    res_pde = res[:, :3] if res.shape[1] >= 3 else res

    u = np.full(xx.size, np.nan)
    v = np.full(xx.size, np.nan)
    p = np.full(xx.size, np.nan)
    speed = np.full(xx.size, np.nan)
    cont = np.full(xx.size, np.nan)

    u[inside] = pred[:, 0]
    v[inside] = pred[:, 1]
    p[inside] = pred[:, 2]
    speed[inside] = np.sqrt(pred[:, 0] ** 2 + pred[:, 1] ** 2)
    cont[inside] = np.abs(res_pde[:, 2])

    return (xx, yy, inside.reshape(xx.shape), u.reshape(xx.shape),
            v.reshape(xx.shape), p.reshape(xx.shape), speed.reshape(xx.shape),
            cont.reshape(xx.shape))


# -------------------------
# 9. Графики
# -------------------------

def draw_geometry(ax):
    poly = np.array(points + [points[0]])
    ax.plot(poly[:, 0], poly[:, 1], linewidth=1.7, color='black')
    ax.set_xlim(0.0, L)
    ax.set_ylim(-H_IN * 1.05, H_IN * 1.05)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")


def preview_geometry(filename=None, n=8000):
    if filename is None:
        filename = os.path.join(OUT_DIR, "geometry_preview.png")
    pts = geom.random_points(n)
    plt.figure(figsize=(9, 3.5))
    plt.scatter(pts[:, 0], pts[:, 1], s=1, alpha=0.5)
    ax = plt.gca()
    draw_geometry(ax)
    ax.set_title("Геометрия: труба с линейным сужением (gPINN)")
    plt.tight_layout()
    plt.savefig(filename, dpi=160)
    plt.close()
    print(f"Сохранено: {filename}")


def plot_training_loss(losshistory, filename=None):
    if filename is None:
        filename = os.path.join(OUT_DIR, "training_loss.png")
    
    train_loss_total = np.sum(np.array(losshistory.loss_train), axis=1)
    test_loss_total = np.sum(np.array(losshistory.loss_test), axis=1)
    steps = np.array(losshistory.steps)
    
    plt.figure(figsize=(8, 4.2))
    plt.semilogy(steps, train_loss_total, 'b-', label='Train loss', linewidth=2)
    plt.semilogy(steps, test_loss_total, 'r--', label='Test loss', linewidth=2)
    plt.xlabel("итерация")
    plt.ylabel("общая ошибка")
    plt.title("Ошибка обучения gPINN")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=160)
    plt.close()
    print(f"Сохранено: {filename}")


def plot_field(field, title, filename, xx, yy, levels=60):
    plt.figure(figsize=(9, 3.5))
    plt.contourf(xx, yy, field, levels=levels, cmap='jet')
    plt.colorbar(label=title)
    ax = plt.gca()
    draw_geometry(ax)
    ax.set_title(f"gPINN: {title}")
    plt.tight_layout()
    plt.savefig(filename, dpi=170)
    plt.close()
    print(f"Сохранено: {filename}")


def plot_streamlines(xx, yy, u, v, speed, filename=None):
    if filename is None:
        filename = os.path.join(OUT_DIR, "streamlines.png")
    plt.figure(figsize=(9, 3.5))
    plt.contourf(xx, yy, speed, levels=60, cmap='jet')
    plt.colorbar(label="speed")
    plt.streamplot(
        xx, yy,
        np.nan_to_num(u, nan=0.0),
        np.nan_to_num(v, nan=0.0),
        density=1.4, linewidth=0.85, color='white', arrowsize=0.8,
    )
    ax = plt.gca()
    draw_geometry(ax)
    ax.set_title("gPINN: Линии тока")
    plt.tight_layout()
    plt.savefig(filename, dpi=170)
    plt.close()
    print(f"Сохранено: {filename}")


def plot_quiver(xx, yy, u, v, speed, filename=None):
    if filename is None:
        filename = os.path.join(OUT_DIR, "quiver.png")
    step = 12
    plt.figure(figsize=(9, 3.5))
    plt.contourf(xx, yy, speed, levels=50, cmap='jet')
    plt.colorbar(label="speed")
    plt.quiver(
        xx[::step, ::step], yy[::step, ::step],
        np.nan_to_num(u[::step, ::step], nan=0.0),
        np.nan_to_num(v[::step, ::step], nan=0.0),
        scale=25, alpha=0.7,
    )
    ax = plt.gca()
    draw_geometry(ax)
    ax.set_title("gPINN: Векторы скорости")
    plt.tight_layout()
    plt.savefig(filename, dpi=170)
    plt.close()
    print(f"Сохранено: {filename}")


def plot_profiles(filename=None, n=250):
    if filename is None:
        filename = os.path.join(OUT_DIR, "profiles.png")

    x = np.linspace(0.0, L, n)[:, None].astype(np.float32)
    center = np.hstack([x, np.zeros_like(x)]).astype(np.float32)
    pred_center = model.predict(center)

    cuts = [0.20, 0.50, 0.80, 1.0, 1.2]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(center[:, 0], pred_center[:, 0], 'b-', label="u(x, 0)")
    axes[0].plot(center[:, 0], pred_center[:, 1], 'r-', label="v(x, 0)")
    axes[0].set_title("gPINN: Скорость по оси трубы")
    axes[0].set_xlabel("x")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    for x_cut in cuts:
        h = float(half_height_numpy(np.array([x_cut]))[0])
        y = np.linspace(-h * 0.98, h * 0.98, n)[:, None].astype(np.float32)
        x_col = np.full_like(y, x_cut, dtype=np.float32)
        pts = np.hstack([x_col, y])
        pred = model.predict(pts)
        axes[1].plot(pred[:, 0], y[:, 0], label=f"x={x_cut:.2f}")

    axes[1].set_title("gPINN: Профили u по вертикальным срезам")
    axes[1].set_xlabel("u")
    axes[1].set_ylabel("y")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=170)
    plt.close()
    print(f"Сохранено: {filename}")


# -------------------------
# 10. Сравнение gPINN с теорией Пуазейля
# -------------------------

def plot_gpinn_vs_poiseuille(filename=None, n_profile=200):
    """
    Сравнение решения gPINN с теоретическим профилем Пуазейля
    на входном и выходном участках канала
    """
    if filename is None:
        filename = os.path.join(OUT_DIR, "gpinn_vs_poiseuille.png")
    
    # Сечения для сравнения
    cuts = [
        (0.1, "x = 0.10 (вход)"),
        (0.5, "x = 0.50 (до сужения)"),
        (1.5, "x = 1.50 (после расширения)"),
        (1.9, "x = 1.90 (выход)")
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()
    
    for idx, (x_cut, title) in enumerate(cuts):
        h_local = float(half_height_numpy(np.array([x_cut]))[0])
        y = np.linspace(-h_local * 0.99, h_local * 0.99, n_profile)[:, None].astype(np.float32)
        x_col = np.full_like(y, x_cut, dtype=np.float32)
        pts = np.hstack([x_col, y])
        
        # gPINN решение
        pred = model.predict(pts)
        u_gpinn = pred[:, 0]
        
        # Теоретический профиль Пуазейля
        u_theory = poiseuille_profile(y.flatten(), h_local, U_MAX)
        
        # Ошибка
        error = np.mean(np.abs(u_gpinn - u_theory))
        
        axes[idx].plot(u_gpinn, y, 'b-', linewidth=2, label=f'gPINN (error={error:.4f})')
        axes[idx].plot(u_theory, y, 'r--', linewidth=2, label='Poiseuille theory')
        axes[idx].set_xlabel('u(x, y)')
        axes[idx].set_ylabel('y')
        axes[idx].set_title(title)
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
        
        # Добавляем границы канала
        axes[idx].axhline(y=h_local, color='k', linestyle='-', alpha=0.5)
        axes[idx].axhline(y=-h_local, color='k', linestyle='-', alpha=0.5)
    
    plt.suptitle('gPINN vs Poiseuille Theory: Профили скорости в различных сечениях', fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=170)
    plt.close()
    print(f"Сохранено: {filename}")
    
    return cuts


def plot_error_analysis(filename=None, n_profile=200):
    """
    Анализ ошибки gPINN относительно теории Пуазейля вдоль канала
    """
    if filename is None:
        filename = os.path.join(OUT_DIR, "error_analysis.png")
    
    x_positions = np.linspace(0.05, L - 0.05, 30)
    errors = []
    
    for x_cut in x_positions:
        h_local = float(half_height_numpy(np.array([x_cut]))[0])
        y = np.linspace(-h_local * 0.95, h_local * 0.95, 50)[:, None].astype(np.float32)
        x_col = np.full_like(y, x_cut, dtype=np.float32)
        pts = np.hstack([x_col, y])
        
        pred = model.predict(pts)
        u_gpinn = pred[:, 0]
        u_theory = poiseuille_profile(y.flatten(), h_local, U_MAX)
        
        # Относительная ошибка
        rel_error = np.mean(np.abs(u_gpinn - u_theory)) / (np.mean(np.abs(u_theory)) + 1e-8)
        errors.append(rel_error)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # График ошибки
    ax1.semilogy(x_positions, errors, 'b-o', markersize=4, linewidth=1.5)
    ax1.axvline(x=X_CONTRACT_1, color='gray', linestyle='--', alpha=0.5, label='Начало сужения')
    ax1.axvline(x=X_CONTRACT_2, color='gray', linestyle='--', alpha=0.5, label='Конец сужения')
    ax1.axvline(x=X_EXPAND_1, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(x=X_EXPAND_2, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('x')
    ax1.set_ylabel('Относительная ошибка')
    ax1.set_title('Ошибка gPINN относительно теории Пуазейля')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Гистограмма ошибки
    ax2.hist(errors, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Относительная ошибка')
    ax2.set_ylabel('Частота')
    ax2.set_title('Распределение ошибки по сечениям')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=170)
    plt.close()
    print(f"Сохранено: {filename}")


def plot_poiseuille_comparison_summary(filename=None, n_profile=150):
    """
    Комплексное сравнение gPINN с теорией Пуазейля
    """
    if filename is None:
        filename = os.path.join(OUT_DIR, "poiseuille_comparison_summary.png")
    
    # Выбираем сечения
    cuts_x = [0.1, 0.5, 1.5, 1.9]
    colors = ['blue', 'green', 'orange', 'red']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for x_cut, color in zip(cuts_x, colors):
        h_local = float(half_height_numpy(np.array([x_cut]))[0])
        y = np.linspace(-h_local * 0.99, h_local * 0.99, n_profile)[:, None].astype(np.float32)
        x_col = np.full_like(y, x_cut, dtype=np.float32)
        pts = np.hstack([x_col, y])
        
        pred = model.predict(pts)
        u_gpinn = pred[:, 0]
        u_theory = poiseuille_profile(y.flatten(), h_local, U_MAX)
        
        ax.plot(u_gpinn, y, color=color, linewidth=2, 
                label=f'gPINN x={x_cut}')
        ax.plot(u_theory, y, color=color, linestyle='--', linewidth=1.5, alpha=0.7)
    
    ax.set_xlabel('u(x, y)')
    ax.set_ylabel('y')
    ax.set_title('gPINN (сплошные) vs Poiseuille Theory (пунктир)\nв различных сечениях канала')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Добавляем информацию о параметрах
    text_str = f'Re = {U_MAX * 2 * H_IN / NU:.1f}\nNU = {NU}\nU_MAX = {U_MAX}'
    ax.text(0.02, 0.98, text_str, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(filename, dpi=170)
    plt.close()
    print(f"Сохранено: {filename}")


# -------------------------
# 11. Диагностика
# -------------------------

def check_model(n_samples=800):
    pts = geom.random_points(n_samples).astype(np.float32)
    pred = model.predict(pts)
    res = _stack_operator_result(model.predict(pts, operator=navier_stokes_with_gradients))
    res_pde = res[:, :3] if res.shape[1] >= 3 else res
    speed = np.sqrt(pred[:, 0] ** 2 + pred[:, 1] ** 2)

    print()
    print("Быстрая диагностика gPINN:")
    print(f"  mean |u|              = {np.mean(np.abs(pred[:, 0])):.4e}")
    print(f"  mean |v|              = {np.mean(np.abs(pred[:, 1])):.4e}")
    print(f"  mean |p|              = {np.mean(np.abs(pred[:, 2])):.4e}")
    print(f"  mean speed            = {np.mean(speed):.4e}")
    print(f"  max speed             = {np.max(speed):.4e}")
    print(f"  mean |mom_x residual| = {np.mean(np.abs(res_pde[:, 0])):.4e}")
    print(f"  mean |mom_y residual| = {np.mean(np.abs(res_pde[:, 1])):.4e}")
    print(f"  mean |continuity|     = {np.mean(np.abs(res_pde[:, 2])):.4e}")
    print(f"  gPINN: +6 gradient terms in loss")


# -------------------------
# 12. Сохранение результатов
# -------------------------

def save_grid_data(filename=None):
    if filename is None:
        filename = os.path.join(OUT_DIR, "grid_data.csv")
    xx, yy, inside, u, v, p, speed, cont = predict_on_grid()
    out = np.column_stack([
        xx.ravel(), yy.ravel(), inside.ravel().astype(int),
        u.ravel(), v.ravel(), p.ravel(), speed.ravel(), cont.ravel(),
    ])
    np.savetxt(filename, out, delimiter=",", 
               header="x,y,inside,u,v,p,speed,continuity_abs", comments="")
    print(f"Сохранено: {filename}")


def create_all_outputs(losshistory):
    xx, yy, inside, u, v, p, speed, cont = predict_on_grid()
    
    plot_training_loss(losshistory)
    plot_field(speed, "Модуль скорости", os.path.join(OUT_DIR, "speed_map.png"), xx, yy)
    plot_field(p, "Давление", os.path.join(OUT_DIR, "pressure_map.png"), xx, yy)
    plot_field(cont, "|Ошибка неразрывности|", os.path.join(OUT_DIR, "continuity_map.png"), xx, yy)
    plot_streamlines(xx, yy, u, v, speed)
    plot_quiver(xx, yy, u, v, speed)
    plot_profiles()
    
    # Дополнительные графики сравнения с теорией Пуазейля
    plot_gpinn_vs_poiseuille()
    plot_error_analysis()
    plot_poiseuille_comparison_summary()
    
    save_grid_data()


# -------------------------
# 13. Запуск
# -------------------------

if __name__ == "__main__":
    preview_geometry()
    losshistory, train_state = train_model()
    check_model()
    create_all_outputs(losshistory)
    print()
    print(f"✅ Все результаты gPINN сохранены в папке: {OUT_DIR}")
    print("📊 gPINN добавляет 6 градиентных членов в функцию потерь")