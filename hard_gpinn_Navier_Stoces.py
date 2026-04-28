import os
import time
import shutil

os.environ.setdefault("DDE_BACKEND", "tensorflow")

import numpy as np
import matplotlib.pyplot as plt
import deepxde as dde
import tensorflow as tf


# ============================================================
# gPINN Navier-Stokes: труба с резким линейным сужением в середине
# Геометрия: прямая труба -> линейное сужение -> узкий участок -> линейное расширение
# ============================================================

# -------------------------
# 1. Настройки
# -------------------------

dde.config.set_random_seed(42)

# Физика
RHO = 1.0
NU = 0.02
U_MAX = 1.0

# Геометрия
L = 2.0
H_IN = 0.50          # половина высоты на входе/выходе
H_THROAT = 0.25      # половина высоты в сужении
X_CONTRACT_1 = 0.75
X_CONTRACT_2 = 0.90
X_EXPAND_1 = 1.10
X_EXPAND_2 = 1.25

# Обучение: увеличенные точки для gPINN
N_DOMAIN = 3000
N_BOUNDARY = 800
N_TEST = 700

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
    """Половина высоты канала h(x), numpy-вариант для графиков/масок."""
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
# 3. gPINN: Уравнения Навье-Стокса + градиенты
# -------------------------

def navier_stokes_with_gradients(x, y):
    """
    gPINN: возвращает невязки PDE и их градиенты по x и y
    """
    u = y[:, 0:1]
    v = y[:, 1:2]
    p = y[:, 2:3]
    
    # Производные первого порядка
    u_x = dde.grad.jacobian(u, x, i=0, j=0)
    u_y = dde.grad.jacobian(u, x, i=0, j=1)
    v_x = dde.grad.jacobian(v, x, i=0, j=0)
    v_y = dde.grad.jacobian(v, x, i=0, j=1)
    p_x = dde.grad.jacobian(p, x, i=0, j=0)
    p_y = dde.grad.jacobian(p, x, i=0, j=1)
    
    # Производные второго порядка
    u_xx = dde.grad.hessian(u, x, i=0, j=0)
    u_yy = dde.grad.hessian(u, x, i=0, j=1)
    v_xx = dde.grad.hessian(v, x, i=0, j=0)
    v_yy = dde.grad.hessian(v, x, i=0, j=1)
    
    # Невязки PDE (конвективная форма Навье-Стокса)
    momentum_x = u * u_x + v * u_y + p_x / RHO - NU * (u_xx + u_yy)
    momentum_y = u * v_x + v * v_y + p_y / RHO - NU * (v_xx + v_yy)
    continuity = u_x + v_y
    
    # ========== gPINN: градиенты невязок ==========
    # Градиенты momentum_x
    g_mom_x_x = dde.grad.jacobian(momentum_x, x, i=0, j=0)
    g_mom_x_y = dde.grad.jacobian(momentum_x, x, i=0, j=1)
    
    # Градиенты momentum_y
    g_mom_y_x = dde.grad.jacobian(momentum_y, x, i=0, j=0)
    g_mom_y_y = dde.grad.jacobian(momentum_y, x, i=0, j=1)
    
    # Градиенты continuity
    g_cont_x = dde.grad.jacobian(continuity, x, i=0, j=0)
    g_cont_y = dde.grad.jacobian(continuity, x, i=0, j=1)
    
    # Возвращаем 9 компонентов: 3 PDE + 6 градиентов
    return [momentum_x, momentum_y, continuity,
            g_mom_x_x, g_mom_x_y, g_mom_y_x, g_mom_y_y, g_cont_x, g_cont_y]


# -------------------------
# 4. Граничные условия
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
    return np.zeros((len(x), 1))


bcs = [
    dde.icbc.DirichletBC(geom, inlet_u, inlet, component=0),
    dde.icbc.DirichletBC(geom, zero, inlet, component=1),
    dde.icbc.DirichletBC(geom, zero, outlet, component=2),
    dde.icbc.DirichletBC(geom, zero, wall, component=0),
    dde.icbc.DirichletBC(geom, zero, wall, component=1),
]


# -------------------------
# 5. Данные и сеть
# -------------------------

data = dde.data.PDE(
    geom,
    navier_stokes_with_gradients,  # gPINN функция
    bcs,
    num_domain=N_DOMAIN,
    num_boundary=N_BOUNDARY,
    num_test=N_TEST,
)

layer_size = [2] + [HIDDEN_WIDTH] * HIDDEN_LAYERS + [3]
net = dde.nn.FNN(layer_size, ACTIVATION, INITIALIZER)
model = dde.Model(data, net)

# Веса для gPINN: 3 PDE + 6 градиентов + 5 BC
# Градиентные члены имеют меньший вес для стабильности
loss_weights = [
    # PDE компоненты
    1.0, 1.0, 15.0,      # momentum_x, momentum_y, continuity
    # Градиенты PDE (gPINN)
    0.1, 0.1,             # d(mom_x)/dx, d(mom_x)/dy
    0.1, 0.1,             # d(mom_y)/dx, d(mom_y)/dy
    0.1, 0.1,             # d(cont)/dx, d(cont)/dy
    # BC компоненты
    20.0, 20.0,           # inlet u, inlet v
    5.0,                  # outlet p
    30.0, 30.0,           # wall u, wall v
]


# -------------------------
# 6. Обучение
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
        model.train()

    return losshistory, train_state


# -------------------------
# 7. Расчет на сетке
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
    pts = np.column_stack([xx.ravel(), yy.ravel()])

    inside = geom.inside(pts)
    pts_inside = pts[inside]
    pred = model.predict(pts_inside)
    res = _stack_operator_result(model.predict(pts_inside, operator=navier_stokes_with_gradients))
    
    # Берём только первые 3 компонента (PDE) для диагностики
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

    return (
        xx,
        yy,
        inside.reshape(xx.shape),
        u.reshape(xx.shape),
        v.reshape(xx.shape),
        p.reshape(xx.shape),
        speed.reshape(xx.shape),
        cont.reshape(xx.shape),
    )


# -------------------------
# 8. Графики
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
    
    # Общая потеря
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
        xx,
        yy,
        np.nan_to_num(u, nan=0.0),
        np.nan_to_num(v, nan=0.0),
        density=1.4,
        linewidth=0.85,
        color='white',
        arrowsize=0.8,
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
        xx[::step, ::step],
        yy[::step, ::step],
        np.nan_to_num(u[::step, ::step], nan=0.0),
        np.nan_to_num(v[::step, ::step], nan=0.0),
        scale=25,
        alpha=0.7,
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

    x = np.linspace(0.0, L, n)[:, None]
    center = np.hstack([x, np.zeros_like(x)])
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
        y = np.linspace(-h * 0.98, h * 0.98, n)[:, None]
        x_col = np.full_like(y, x_cut)
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
# 9. Диагностика
# -------------------------

def check_model(n_samples=800):
    pts = geom.random_points(n_samples)
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
# 10. Сохранение результатов
# -------------------------

def save_grid_data(filename=None):
    if filename is None:
        filename = os.path.join(OUT_DIR, "grid_data.csv")
    xx, yy, inside, u, v, p, speed, cont = predict_on_grid()
    out = np.column_stack([
        xx.ravel(), yy.ravel(), inside.ravel().astype(int),
        u.ravel(), v.ravel(), p.ravel(), speed.ravel(), cont.ravel(),
    ])
    np.savetxt(
        filename,
        out,
        delimiter=",",
        header="x,y,inside,u,v,p,speed,continuity_abs",
        comments="",
    )
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
    save_grid_data()


# -------------------------
# 11. Запуск
# -------------------------

if __name__ == "__main__":
    preview_geometry()
    losshistory, train_state = train_model()
    check_model()
    create_all_outputs(losshistory)
    print()
    print(f"✅ Все результаты gPINN сохранены в папке: {OUT_DIR}")
    print("📊 gPINN добавляет 6 градиентных членов в функцию потерь")