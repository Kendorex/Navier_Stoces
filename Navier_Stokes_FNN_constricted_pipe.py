import os
import time
import shutil

os.environ.setdefault("DDE_BACKEND", "tensorflow")

import numpy as np
import matplotlib.pyplot as plt
import deepxde as dde


# ============================================================
# PINN Navier-Stokes: труба с резким линейным сужением в середине
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

# Обучение: быстрый режим для дебага
N_DOMAIN = 2500
N_BOUNDARY = 700
N_TEST = 600

HIDDEN_LAYERS = 4
HIDDEN_WIDTH = 56
ACTIVATION = "tanh"
INITIALIZER = "Glorot uniform"

LR_ADAM = 1e-3
ITERATIONS_ADAM = 2500
DISPLAY_EVERY = 250
USE_LBFGS = False

# Вывод
MODEL_NAME = "navier_stokes_constricted_pipe"
OUT_DIR = f"{MODEL_NAME}_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

GRID_NX = 420
GRID_NY = 140


# -------------------------
# 2. Геометрия
# -------------------------

# Верхняя стенка идет слева направо, нижняя справа налево.
# Сужение сделано прямыми отрезками, не плавной кривой.
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
# 3. Уравнения Навье-Стокса
# -------------------------

def pde(x, y):
    u = y[:, 0:1]
    v = y[:, 1:2]

    u_x = dde.grad.jacobian(y, x, i=0, j=0)
    u_y = dde.grad.jacobian(y, x, i=0, j=1)
    v_x = dde.grad.jacobian(y, x, i=1, j=0)
    v_y = dde.grad.jacobian(y, x, i=1, j=1)

    p_x = dde.grad.jacobian(y, x, i=2, j=0)
    p_y = dde.grad.jacobian(y, x, i=2, j=1)

    u_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    u_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)
    v_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)
    v_yy = dde.grad.hessian(y, x, component=1, i=1, j=1)

    momentum_x = u * u_x + v * u_y + p_x / RHO - NU * (u_xx + u_yy)
    momentum_y = u * v_x + v * v_y + p_y / RHO - NU * (v_xx + v_yy)
    continuity = u_x + v_y

    return [momentum_x, momentum_y, continuity]


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
    # Вход: задаем профиль скорости
    dde.icbc.DirichletBC(geom, inlet_u, inlet, component=0),
    dde.icbc.DirichletBC(geom, zero, inlet, component=1),

    # Выход: фиксируем только давление. Скорость на выходе не зажимаем.
    dde.icbc.DirichletBC(geom, zero, outlet, component=2),

    # Стенки: no-slip
    dde.icbc.DirichletBC(geom, zero, wall, component=0),
    dde.icbc.DirichletBC(geom, zero, wall, component=1),
]


# -------------------------
# 5. Данные и сеть
# -------------------------

data = dde.data.PDE(
    geom,
    pde,
    bcs,
    num_domain=N_DOMAIN,
    num_boundary=N_BOUNDARY,
    num_test=N_TEST,
)

layer_size = [2] + [HIDDEN_WIDTH] * HIDDEN_LAYERS + [3]
net = dde.nn.FNN(layer_size, ACTIVATION, INITIALIZER)
model = dde.Model(data, net)

# Порядок: 3 PDE loss + 5 BC loss
loss_weights = [
    1.0, 1.0, 15.0,   # momentum_x, momentum_y, continuity
    20.0, 20.0,       # inlet u, inlet v
    5.0,              # outlet p
    30.0, 30.0,       # wall u, wall v
]


# -------------------------
# 6. Обучение
# -------------------------

def train_model():
    print("=" * 60)
    print("PINN: труба с линейным сужением в середине")
    print("=" * 60)
    print(f"NU={NU}, U_MAX={U_MAX}, H_IN={H_IN}, H_THROAT={H_THROAT}, L={L}")
    print(f"Точки: внутри={N_DOMAIN}, граница={N_BOUNDARY}, тест={N_TEST}")
    print(f"Сеть: {layer_size}")
    print(f"Adam итерации: {ITERATIONS_ADAM}")
    print(f"L-BFGS: {USE_LBFGS}")
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
    res = _stack_operator_result(model.predict(pts_inside, operator=pde))

    u = np.full(xx.size, np.nan)
    v = np.full(xx.size, np.nan)
    p = np.full(xx.size, np.nan)
    speed = np.full(xx.size, np.nan)
    cont = np.full(xx.size, np.nan)

    u[inside] = pred[:, 0]
    v[inside] = pred[:, 1]
    p[inside] = pred[:, 2]
    speed[inside] = np.sqrt(pred[:, 0] ** 2 + pred[:, 1] ** 2)
    cont[inside] = np.abs(res[:, 2])

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
    ax.plot(poly[:, 0], poly[:, 1], linewidth=1.7)
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
    plt.scatter(pts[:, 0], pts[:, 1], s=1)
    ax = plt.gca()
    draw_geometry(ax)
    ax.set_title("Геометрия: труба с линейным сужением")
    plt.tight_layout()
    plt.savefig(filename, dpi=160)
    plt.close()
    print(f"Сохранено: {filename}")


def plot_training_loss(losshistory, filename=None):
    if filename is None:
        filename = os.path.join(OUT_DIR, "training_loss.png")
    train_loss = np.sum(np.array(losshistory.loss_train), axis=1)
    steps = np.array(losshistory.steps)
    plt.figure(figsize=(8, 4.2))
    plt.semilogy(steps, train_loss)
    plt.xlabel("итерация")
    plt.ylabel("общая ошибка")
    plt.title("Ошибка обучения")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=160)
    plt.close()
    print(f"Сохранено: {filename}")


def plot_field(field, title, filename, xx, yy, levels=60):
    plt.figure(figsize=(9, 3.5))
    plt.contourf(xx, yy, field, levels=levels)
    plt.colorbar(label=title)
    ax = plt.gca()
    draw_geometry(ax)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=170)
    plt.close()
    print(f"Сохранено: {filename}")


def plot_streamlines(xx, yy, u, v, speed, filename=None):
    if filename is None:
        filename = os.path.join(OUT_DIR, "streamlines.png")
    plt.figure(figsize=(9, 3.5))
    plt.contourf(xx, yy, speed, levels=60)
    plt.colorbar(label="speed")
    plt.streamplot(
        xx,
        yy,
        np.nan_to_num(u, nan=0.0),
        np.nan_to_num(v, nan=0.0),
        density=1.4,
        linewidth=0.85,
        arrowsize=0.8,
    )
    ax = plt.gca()
    draw_geometry(ax)
    ax.set_title("Линии тока")
    plt.tight_layout()
    plt.savefig(filename, dpi=170)
    plt.close()
    print(f"Сохранено: {filename}")


def plot_quiver(xx, yy, u, v, speed, filename=None):
    if filename is None:
        filename = os.path.join(OUT_DIR, "quiver.png")
    step = 8
    plt.figure(figsize=(9, 3.5))
    plt.contourf(xx, yy, speed, levels=50)
    plt.colorbar(label="speed")
    plt.quiver(
        xx[::step, ::step],
        yy[::step, ::step],
        np.nan_to_num(u[::step, ::step], nan=0.0),
        np.nan_to_num(v[::step, ::step], nan=0.0),
        scale=28,
    )
    ax = plt.gca()
    draw_geometry(ax)
    ax.set_title("Векторы скорости")
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

    # вертикальные срезы: до сужения, в горле, после расширения
    cuts = [0.20, 0.50, 0.80]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(center[:, 0], pred_center[:, 0], label="u(x, 0)")
    axes[0].plot(center[:, 0], pred_center[:, 1], label="v(x, 0)")
    axes[0].set_title("Скорость по оси трубы")
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

    axes[1].set_title("Профили u по вертикальным срезам")
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
    res = _stack_operator_result(model.predict(pts, operator=pde))
    speed = np.sqrt(pred[:, 0] ** 2 + pred[:, 1] ** 2)

    print()
    print("Быстрая диагностика:")
    print(f"  mean |u|              = {np.mean(np.abs(pred[:, 0])):.4e}")
    print(f"  mean |v|              = {np.mean(np.abs(pred[:, 1])):.4e}")
    print(f"  mean |p|              = {np.mean(np.abs(pred[:, 2])):.4e}")
    print(f"  mean speed            = {np.mean(speed):.4e}")
    print(f"  max speed             = {np.max(speed):.4e}")
    print(f"  mean |mom_x residual| = {np.mean(np.abs(res[:, 0])):.4e}")
    print(f"  mean |mom_y residual| = {np.mean(np.abs(res[:, 1])):.4e}")
    print(f"  mean |continuity|     = {np.mean(np.abs(res[:, 2])):.4e}")


# -------------------------
# 10. GIF-анимация без мусора в проекте
# -------------------------

def create_flow_animation(filename=None, n_frames=160):
    import imageio.v2 as imageio

    if filename is None:
        filename = os.path.join(OUT_DIR, "flow.gif")

    frames_dir = os.path.join(OUT_DIR, "_animation_frames")
    os.makedirs(frames_dir, exist_ok=True)

    print("Создание GIF-анимации...")

    rng = np.random.default_rng(42)
    n_particles = 320
    dt = 0.007

    def random_particles(n):
        pts = []

        while len(pts) < n:
            x_new = rng.uniform(0.0, L, n * 3)
            h_new = half_height_numpy(x_new)
            y_new = rng.uniform(-0.90 * h_new, 0.90 * h_new)

            cand = np.column_stack([x_new, y_new])
            cand = cand[geom.inside(cand)]

            pts.extend(cand.tolist())

        return np.array(pts[:n])

    particles = random_particles(n_particles)

    def reset_particles(indices):
        if len(indices) > 0:
            particles[indices] = random_particles(len(indices))

    # Фоновое поле скорости
    xs = np.linspace(0.0, L, 240)
    ys = np.linspace(-H_IN, H_IN, 120)
    xx, yy = np.meshgrid(xs, ys)

    grid_pts = np.column_stack([xx.ravel(), yy.ravel()])
    inside = geom.inside(grid_pts).reshape(xx.shape)

    pred_grid = model.predict(grid_pts)
    u_grid = pred_grid[:, 0].reshape(xx.shape)
    v_grid = pred_grid[:, 1].reshape(xx.shape)
    speed = np.sqrt(u_grid ** 2 + v_grid ** 2)

    speed_masked = np.where(inside, speed, np.nan)

    poly = np.array(points + [points[0]])
    frame_paths = []

    for frame in range(n_frames):
        vel = model.predict(particles)

        # Защита от единичных слишком больших скачков
        vel_norm = np.sqrt(vel[:, 0] ** 2 + vel[:, 1] ** 2)
        max_vel = np.nanpercentile(vel_norm, 98)

        if max_vel > 1e-8:
            scale = np.minimum(1.0, max_vel / (vel_norm + 1e-8))
            vel[:, 0] *= scale
            vel[:, 1] *= scale

        particles[:, 0] += vel[:, 0] * dt
        particles[:, 1] += vel[:, 1] * dt

        h_particles = half_height_numpy(particles[:, 0])

        too_close_to_wall = np.abs(particles[:, 1]) > 0.97 * h_particles

        valid = (
            (particles[:, 0] >= 0.0)
            & (particles[:, 0] <= L)
            & geom.inside(particles)
            & (~too_close_to_wall)
        )

        bad = np.where(~valid)[0]
        reset_particles(bad)

        fig, ax = plt.subplots(figsize=(10, 3.2))

        cf = ax.contourf(
            xx,
            yy,
            speed_masked,
            levels=50,
            alpha=0.85,
        )
        plt.colorbar(cf, ax=ax, label="speed")

        ax.scatter(
            particles[:, 0],
            particles[:, 1],
            s=8,
            alpha=0.85,
        )

        ax.plot(poly[:, 0], poly[:, 1], linewidth=2.0)

        ax.set_xlim(0.0, L)
        ax.set_ylim(-H_IN * 1.05, H_IN * 1.05)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Маркеры потока в поле скорости")

        plt.tight_layout()

        frame_path = os.path.join(frames_dir, f"frame_{frame:04d}.png")
        plt.savefig(frame_path, dpi=120)
        plt.close(fig)

        frame_paths.append(frame_path)

    frames = [imageio.imread(path) for path in frame_paths]
    imageio.mimsave(filename, frames, duration=0.055)

    shutil.rmtree(frames_dir, ignore_errors=True)
    print(f"GIF сохранена: {filename}")

# -------------------------
# 11. Сохранение результатов
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
    create_flow_animation()


# -------------------------
# 12. Запуск
# -------------------------

if __name__ == "__main__":
    preview_geometry()
    losshistory, train_state = train_model()
    check_model()
    create_all_outputs(losshistory)
    print()
    print(f"Все результаты сохранены в папке: {OUT_DIR}")
