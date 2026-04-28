import deepxde as dde
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from datetime import datetime

output_dir = "output_stokes_hard"
os.makedirs(output_dir, exist_ok=True)
run_dir = os.path.join(output_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
os.makedirs(run_dir)

# Геометрия (как на картинке)
L, H = 2.0, 0.5
geom = dde.geometry.Rectangle([0, 0], [L, H])

# Границы
def inflow(x, on): return on and np.isclose(x[0], 0)
def outflow(x, on): return on and np.isclose(x[0], L)
def wall(x, on): return on and (np.isclose(x[1], 0) or np.isclose(x[1], H))

def inlet_u(x):
    y = x[:, 1:2]
    u_max = 1.0
    return 4 * u_max * (y / H) * (1 - y / H)

bc_inlet_u = dde.icbc.DirichletBC(geom, inlet_u, inflow, component=0)
bc_inlet_v = dde.icbc.DirichletBC(geom, lambda x: 0.0, inflow, component=1)
bc_wall_u = dde.icbc.DirichletBC(geom, lambda x: 0.0, wall, component=0)
bc_wall_v = dde.icbc.DirichletBC(geom, lambda x: 0.0, wall, component=1)
bc_out_p = dde.icbc.DirichletBC(geom, lambda x: 0.0, outflow, component=2)

# Уравнения Стокса (сильная форма)
def stokes_pde(x, y):
    u = y[:, 0:1]
    v = y[:, 1:2]
    p = y[:, 2:3]
    mu = 1.0
    
    u_x = dde.grad.jacobian(u, x, i=0, j=0)
    u_y = dde.grad.jacobian(u, x, i=0, j=1)
    v_x = dde.grad.jacobian(v, x, i=0, j=0)
    v_y = dde.grad.jacobian(v, x, i=0, j=1)
    p_x = dde.grad.jacobian(p, x, i=0, j=0)
    p_y = dde.grad.jacobian(p, x, i=0, j=1)
    
    u_xx = dde.grad.hessian(u, x, i=0, j=0)
    u_yy = dde.grad.hessian(u, x, i=0, j=1)
    v_xx = dde.grad.hessian(v, x, i=0, j=0)
    v_yy = dde.grad.hessian(v, x, i=0, j=1)
    
    momentum_x = -p_x + mu * (u_xx + u_yy)
    momentum_y = -p_y + mu * (v_xx + v_yy)
    continuity = u_x + v_y
    
    return [momentum_x, momentum_y, continuity]

data = dde.data.PDE(
    geom, stokes_pde,
    [bc_inlet_u, bc_inlet_v, bc_wall_u, bc_wall_v, bc_out_p],
    num_domain=1000,
    num_boundary=200,
    num_test=200
)

layer_size = [2] + [64, 64, 64] + [3]
net = dde.nn.FNN(layer_size, "tanh", "Glorot uniform")
model = dde.Model(data, net)
model.compile("adam", lr=0.001, loss="MSE")

print("Обучение модели течения в канале...")
losshistory, train_state = model.train(iterations=3000, display_every=500)

# ========== ГРАФИК КАК НА КАРТИНКЕ ==========
nx, ny = 80, 40
x = np.linspace(0, L, nx)
y = np.linspace(0, H, ny)
X, Y = np.meshgrid(x, y)
points = np.hstack((X.reshape(-1,1), Y.reshape(-1,1)))

output = model.predict(points)
U = output[:, 0].reshape(nx, ny)
V = output[:, 1].reshape(nx, ny)

# Вычисляем модуль скорости
speed = np.sqrt(U**2 + V**2)

# ГРАФИК 1: Линии тока + цветовая карта (как на картинке)
fig, ax = plt.subplots(1, 1, figsize=(14, 4))

# Цветовая карта скорости
cf = ax.contourf(X, Y, speed.T, levels=30, cmap='jet')
cbar = plt.colorbar(cf, ax=ax, label='Скорость, м/с', pad=0.01)

# Линии тока (без alpha, вместо этого используем linewidth)
ax.streamplot(X, Y, U.T, V.T, color='white', linewidth=0.8, density=2)

# Оформление
ax.set_xlabel('x, м', fontsize=12)
ax.set_ylabel('y, м', fontsize=12)
ax.set_title('Линии тока и поле скорости в трещине', fontsize=14, fontweight='bold')
ax.set_xlim(0, L)
ax.set_ylim(0, H)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3, linestyle='--')

# Метки входа и выхода
ax.text(0.02, 0.5, 'ВХОД', transform=ax.transAxes, fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
ax.text(0.95, 0.5, 'ВЫХОД', transform=ax.transAxes, fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

plt.tight_layout()
plt.savefig(os.path.join(run_dir, 'channel_flow.png'), dpi=200, bbox_inches='tight')
plt.show()

# ГРАФИК 2: Профиль скорости
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

y_profile = np.linspace(0, H, 100)
x_mid = 1.0
pts_mid = np.array([[x_mid, y] for y in y_profile])
u_mid = model.predict(pts_mid)[:, 0]
u_theory = 4 * (y_profile / H) * (1 - y_profile / H)

axes[0].plot(u_mid, y_profile, 'b-', linewidth=2.5, label='PINN решение')
axes[0].plot(u_theory, y_profile, 'r--', linewidth=2, label='Теория Пуазейля')
axes[0].set_xlabel('Скорость u, м/с', fontsize=11)
axes[0].set_ylabel('y, м', fontsize=11)
axes[0].set_title('Профиль скорости в центре канала (x=1.0)', fontsize=11)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# ГРАФИК 3: Сохранение расхода
x_sections = np.linspace(0.2, 1.8, 10)
flows = []
for xs in x_sections:
    pts = np.array([[xs, y] for y in y_profile])
    u_vals = model.predict(pts)[:, 0]
    flows.append(np.trapz(u_vals, y_profile))

axes[1].plot(x_sections, flows, 'bo-', linewidth=2, markersize=6)
axes[1].axhline(y=2*H/3, color='r', linestyle='--', label=f'Теория: {2*H/3:.4f}')
axes[1].set_xlabel('x, м', fontsize=11)
axes[1].set_ylabel('Расход, м³/с', fontsize=11)
axes[1].set_title('Сохранение массы вдоль канала', fontsize=11)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(run_dir, 'additional_plots.png'), dpi=150, bbox_inches='tight')
plt.show()

# Результаты
print("\n" + "="*50)
print("РЕЗУЛЬТАТЫ")
print("="*50)
print(f"Размеры: L={L} м, H={H} м")
print(f"Средний расход: {np.mean(flows):.4f} м³/с (теория: {2*H/3:.4f})")
print(f"Ошибка: {abs(np.mean(flows)-2*H/3)/(2*H/3)*100:.2f}%")
print(f"Сохранение массы: {(flows[-1]-flows[0])/flows[0]*100:.2f}%")
print("="*50)
print(f"✅ Результаты сохранены в: {run_dir}")