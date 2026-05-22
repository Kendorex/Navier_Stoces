import deepxde as dde
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from datetime import datetime

output_dir = "output_gpinn_stokes"
os.makedirs(output_dir, exist_ok=True)
run_dir = os.path.join(output_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
os.makedirs(run_dir)

L, H = 2.0, 0.5
geom = dde.geometry.Rectangle([0, 0], [L, H])

def inflow(x, on): return on and np.isclose(x[0], 0)
def outflow(x, on): return on and np.isclose(x[0], L)
def wall(x, on): return on and (np.isclose(x[1], 0) or np.isclose(x[1], H))

def inlet_u(x):
    y = x[:, 1:2]
    u_max = 1.0
    return 4 * u_max * (y / H) * (1 - y / H)

# Граничные условия
bc_inlet_u = dde.icbc.DirichletBC(geom, inlet_u, inflow, component=0)
bc_inlet_v = dde.icbc.DirichletBC(geom, lambda x: 0.0, inflow, component=1)
bc_wall_u = dde.icbc.DirichletBC(geom, lambda x: 0.0, wall, component=0)
bc_wall_v = dde.icbc.DirichletBC(geom, lambda x: 0.0, wall, component=1)

# Фиксируем давление в точке (а не на всей границе)
bc_out_p = dde.icbc.PointSetBC(
    np.array([[L, H/2]]),
    np.array([[0.0]]),
    component=2
)

def stokes_pde_with_gradients(x, y):
    """gPINN: уравнения Стокса + градиенты невязок"""
    u = y[:, 0:1]
    v = y[:, 1:2]
    p = y[:, 2:3]
    
    # Градиенты первого порядка
    u_x = dde.grad.jacobian(u, x, i=0, j=0)
    u_y = dde.grad.jacobian(u, x, i=0, j=1)
    v_x = dde.grad.jacobian(v, x, i=0, j=0)
    v_y = dde.grad.jacobian(v, x, i=0, j=1)
    p_x = dde.grad.jacobian(p, x, i=0, j=0)
    p_y = dde.grad.jacobian(p, x, i=0, j=1)
    
    # Вторые производные
    u_xx = dde.grad.hessian(u, x, i=0, j=0)
    u_yy = dde.grad.hessian(u, x, i=0, j=1)
    v_xx = dde.grad.hessian(v, x, i=0, j=0)
    v_yy = dde.grad.hessian(v, x, i=0, j=1)
    
    # Уравнения Стокса
    momentum_x = -p_x + u_xx + u_yy
    momentum_y = -p_y + v_xx + v_yy
    continuity = u_x + v_y
    
    # Градиенты невязок для gPINN
    g_mom_x_x = dde.grad.jacobian(momentum_x, x, i=0, j=0)
    g_mom_x_y = dde.grad.jacobian(momentum_x, x, i=0, j=1)
    g_mom_y_x = dde.grad.jacobian(momentum_y, x, i=0, j=0)
    g_mom_y_y = dde.grad.jacobian(momentum_y, x, i=0, j=1)
    g_cont_x = dde.grad.jacobian(continuity, x, i=0, j=0)
    g_cont_y = dde.grad.jacobian(continuity, x, i=0, j=1)
    
    return [momentum_x, momentum_y, continuity,
            g_mom_x_x, g_mom_x_y, g_mom_y_x, g_mom_y_y, g_cont_x, g_cont_y]


data = dde.data.PDE(
    geom, 
    stokes_pde_with_gradients,
    [bc_inlet_u, bc_inlet_v, bc_wall_u, bc_wall_v, bc_out_p],
    num_domain=4000,
    num_boundary=800,
    num_test=1000
)

# Сеть побольше для gPINN
layer_size = [2] + [128, 128, 128, 128] + [3]
net = dde.nn.FNN(layer_size, "tanh", "Glorot normal")

model = dde.Model(data, net)

# Ключевой момент: веса для разных компонент loss
# Основные уравнения: вес 1
# Градиенты невязок: вес 0.01 (меньше, чтобы не доминировали)
# ГУ: вес 10
# PointSet: вес 100
model.compile("adam", lr=0.001, 
              loss_weights=[1, 1, 1,           # momentum_x, momentum_y, continuity
                          0.01, 0.01, 0.01, 0.01, 0.01, 0.01,  # градиенты
                          10, 10, 10, 10,     # ГУ: u_in, v_in, u_wall, v_wall
                          100])                # PointSet p=0 (высокий вес!)

print("Обучение gPINN модели Стокса (L={}, H={})...".format(L, H))
losshistory, train_state = model.train(iterations=8000, display_every=1000)


def analytical_solution(x, y):
    """Аналитическое решение для течения Пуазейля"""
    u_max = 1.0
    mu = 1.0
    u = 4 * u_max * (y / H) * (1 - y / H)
    v = np.zeros_like(x)
    dp_dx = -8.0 * mu * u_max / (H**2)  # = -32
    p = dp_dx * (x - L)  # p(L) = 0
    return u, v, p


def plot_training_loss(losshistory, filename):
    """График истории обучения"""
    train_loss_total = np.sum(np.array(losshistory.loss_train), axis=1)
    test_loss_total = np.sum(np.array(losshistory.loss_test), axis=1)
    steps = np.array(losshistory.steps)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(steps, train_loss_total, label='Train loss', linewidth=2)
    ax.semilogy(steps, test_loss_total, label='Test loss', linewidth=2, alpha=0.7)
    ax.legend(fontsize=12)
    ax.set_title('gPINN Training History', fontsize=14, fontweight='bold')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def plot_results_comprehensive(filename):
    """Комплексная визуализация с истинными и предсказанными полями"""
    GRID_NX, GRID_NY = 200, 80
    xs = np.linspace(0, L, GRID_NX, dtype=np.float32)
    ys = np.linspace(0, H, GRID_NY, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)
    pts = np.column_stack([X.ravel(), Y.ravel()]).astype(np.float32)
    
    # Предсказания модели
    output = model.predict(pts)
    u_pred = output[:, 0].reshape(GRID_NY, GRID_NX)
    v_pred = output[:, 1].reshape(GRID_NY, GRID_NX)
    p_pred = output[:, 2].reshape(GRID_NY, GRID_NX)
    
    # Аналитическое решение
    u_true, v_true, p_true = analytical_solution(X, Y)
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    
    # u-скорость (истинная)
    im1 = axes[0, 0].contourf(X, Y, u_true, levels=20, cmap='RdBu_r')
    axes[0, 0].set_title('u-velocity (True)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # u-скорость (предсказанная)
    im2 = axes[0, 1].contourf(X, Y, u_pred, levels=20, cmap='RdBu_r')
    axes[0, 1].set_title('u-velocity (Predicted)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Давление (истинное)
    im3 = axes[0, 2].contourf(X, Y, p_true, levels=20, cmap='viridis')
    axes[0, 2].set_title('Pressure (True)', fontsize=12, fontweight='bold')
    axes[0, 2].set_xlabel('x')
    axes[0, 2].set_ylabel('y')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # Давление (предсказанное)
    im4 = axes[0, 3].contourf(X, Y, p_pred, levels=20, cmap='viridis')
    axes[0, 3].set_title('Pressure (Predicted)', fontsize=12, fontweight='bold')
    axes[0, 3].set_xlabel('x')
    axes[0, 3].set_ylabel('y')
    plt.colorbar(im4, ax=axes[0, 3])
    
    # Профиль скорости в центре
    mid_x = GRID_NX // 2
    axes[1, 0].plot(u_true[:, mid_x], ys, 'b-', label='True', linewidth=2.5)
    axes[1, 0].plot(u_pred[:, mid_x], ys, 'r--', label='Predicted', linewidth=2.5)
    axes[1, 0].set_title('Velocity Profile at x=L/2', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('u-velocity', fontsize=11)
    axes[1, 0].set_ylabel('y', fontsize=11)
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Профиль скорости на выходе
    axes[1, 1].plot(u_true[:, -1], ys, 'b-', label='True', linewidth=2.5)
    axes[1, 1].plot(u_pred[:, -1], ys, 'r--', label='Predicted', linewidth=2.5)
    axes[1, 1].set_title('Velocity Profile at x=L', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('u-velocity', fontsize=11)
    axes[1, 1].set_ylabel('y', fontsize=11)
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Поле скорости (векторное)
    skip = max(1, min(GRID_NX, GRID_NY) // 25)
    axes[1, 2].quiver(X[::skip, ::skip], Y[::skip, ::skip], 
                   u_pred[::skip, ::skip], v_pred[::skip, ::skip],
                   scale=50, width=0.003)
    axes[1, 2].set_title('Predicted Velocity Field', fontsize=12, fontweight='bold')
    axes[1, 2].set_xlabel('x', fontsize=11)
    axes[1, 2].set_ylabel('y', fontsize=11)
    axes[1, 2].set_aspect('equal')
    axes[1, 2].grid(True, alpha=0.3)
    
    # Давление по центру
    axes[1, 3].plot(xs, p_true[0, :], 'b-', label='True', linewidth=2.5)
    axes[1, 3].plot(xs, p_pred[0, :], 'r--', label='Predicted', linewidth=2.5)
    axes[1, 3].set_title('Pressure along y=0', fontsize=12, fontweight='bold')
    axes[1, 3].set_xlabel('x', fontsize=11)
    axes[1, 3].set_ylabel('Pressure', fontsize=11)
    axes[1, 3].legend(fontsize=10)
    axes[1, 3].grid(True, alpha=0.3)
    
    plt.suptitle('gPINN: Stokes Flow in Channel', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Сохранено: {filename}")


# Создание графиков
plot_training_loss(losshistory, os.path.join(run_dir, 'training_loss.png'))
plot_results_comprehensive(os.path.join(run_dir, 'comprehensive_results.png'))

# Проверка расхода
y_sec = np.linspace(0, H, 50)
print("\nMass flow rate through sections (gPINN):")
flows = []
for x0 in [0.25, 0.5, 0.75]:
    pts = np.array([[x0, y] for y in y_sec]).astype(np.float32)
    u_vals = model.predict(pts)[:, 0]
    flow = np.trapz(u_vals, y_sec)
    flows.append(flow)
    print(f"  x = {x0}: {flow:.4f}")

theoretical_flow = 2 * H / 3

# Расчет ошибок
GRID_NX, GRID_NY = 200, 80
xs = np.linspace(0, L, GRID_NX, dtype=np.float32)
ys = np.linspace(0, H, GRID_NY, dtype=np.float32)
X, Y = np.meshgrid(xs, ys)
pts = np.column_stack([X.ravel(), Y.ravel()]).astype(np.float32)
output = model.predict(pts)
u_pred = output[:, 0]
v_pred = output[:, 1]
p_pred = output[:, 2]

u_true, v_true, p_true = analytical_solution(X.flatten(), Y.flatten())

# Ошибки скорости
rel_error_u = np.linalg.norm(u_pred - u_true) / np.linalg.norm(u_true)
rel_error_v = np.linalg.norm(v_pred - v_true) / max(np.linalg.norm(u_true), 1e-10)
mse_u = np.mean((u_pred - u_true)**2)
mse_v = np.mean((v_pred - v_true)**2)
max_error_u = np.max(np.abs(u_pred - u_true))
max_error_v = np.max(np.abs(v_pred - v_true))

# Ошибки давления
rel_error_p = np.linalg.norm(p_pred - p_true) / np.linalg.norm(p_true)
mse_p = np.mean((p_pred - p_true)**2)
max_error_p = np.max(np.abs(p_pred - p_true))

# Градиент давления (физически важная величина)
dp_dx_true = -8.0 * 1.0 / (H**2)  # = -32
dp_pred_2d = p_pred.reshape(GRID_NY, GRID_NX)
dp_dx_pred = np.gradient(dp_pred_2d, xs, axis=1)
dp_dx_pred_mean = np.mean(dp_dx_pred)

# Проверка давления в точке закрепления
p_at_outlet = model.predict(np.array([[L, H/2]]).astype(np.float32))[0, 2]

print("\n" + "="*60)
print("gPINN RESULTS SUMMARY")
print("="*60)
print(f"Channel: L={L}, H={H} (aspect ratio={L/H:.0f}:1)")

if isinstance(losshistory.loss_train[-1], np.ndarray):
    final_loss_total = np.sum(losshistory.loss_train[-1])
    print(f"Final loss: {final_loss_total:.6f}")
else:
    print(f"Final loss: {losshistory.loss_train[-1]:.6f}")

print(f"\nОшибки скорости:")
print(f"  u-velocity (relative): {rel_error_u:.4f} ({rel_error_u*100:.2f}%)")
print(f"  v-velocity (relative): {rel_error_v:.6f}")
print(f"  u-velocity (MSE):      {mse_u:.6f}")
print(f"  v-velocity (MSE):      {mse_v:.6f}")
print(f"  u-velocity (max):      {max_error_u:.4f}")
print(f"  v-velocity (max):      {max_error_v:.4f}")

print(f"\nОшибки давления:")
print(f"  relative: {rel_error_p:.4f} ({rel_error_p*100:.2f}%)")
print(f"  MSE:      {mse_p:.4f}")
print(f"  max:      {max_error_p:.4f}")

print(f"\nГрадиент давления:")
print(f"  Истинный dp/dx:     {dp_dx_true:.4f}")
print(f"  Предсказанный dp/dx: {dp_dx_pred_mean:.4f}")
print(f"  Ошибка градиента:   {abs(dp_dx_pred_mean - dp_dx_true)/abs(dp_dx_true)*100:.2f}%")

print(f"\nДавление в точке закрепления (должно быть 0):")
print(f"  p(L, H/2) = {p_at_outlet:.6f}")

print(f"\nРасход:")
print(f"  Avg flow rate: {np.mean(flows):.4f} (theoretical: {theoretical_flow:.4f})")
print(f"  Error: {abs(np.mean(flows)-theoretical_flow)/theoretical_flow*100:.2f}%")
print(f"\nВсе результаты сохранены в: {run_dir}")