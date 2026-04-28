import deepxde as dde
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Папка для результатов
output_dir = "output_stokes"
os.makedirs(output_dir, exist_ok=True)
run_dir = os.path.join(output_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
os.makedirs(run_dir)

# Геометрия
L, H = 2.0, 0.5
geom = dde.geometry.Rectangle([0, 0], [L, H])

# Границы
def inflow(x, on): return on and np.isclose(x[0], 0)
def outflow(x, on): return on and np.isclose(x[0], L)
def wall(x, on): return on and (np.isclose(x[1], 0) or np.isclose(x[1], H))

# Входной профиль
def inlet_u(x):
    y = x[:, 1:2]
    return 4 * (y / H) * (1 - y / H)

# ГУ
bc_inlet_u = dde.icbc.DirichletBC(geom, inlet_u, inflow, component=0)
bc_inlet_v = dde.icbc.DirichletBC(geom, lambda x: 0.0, inflow, component=1)
bc_wall_u = dde.icbc.DirichletBC(geom, lambda x: 0.0, wall, component=0)
bc_wall_v = dde.icbc.DirichletBC(geom, lambda x: 0.0, wall, component=1)
bc_out_p = dde.icbc.DirichletBC(geom, lambda x: 0.0, outflow, component=2)

# Уравнения Стокса
def stokes_pde(x, y):
    u = y[:, 0:1]
    v = y[:, 1:2]
    p = y[:, 2:3]
    mu = 1.0
    
    u_x = dde.grad.jacobian(u, x, i=0, j=0)
    u_y = dde.grad.jacobian(u, x, i=0, j=1)
    u_xx = dde.grad.hessian(u, x, i=0, j=0)
    u_yy = dde.grad.hessian(u, x, i=0, j=1)
    
    v_x = dde.grad.jacobian(v, x, i=0, j=0)
    v_y = dde.grad.jacobian(v, x, i=0, j=1)
    v_xx = dde.grad.hessian(v, x, i=0, j=0)
    v_yy = dde.grad.hessian(v, x, i=0, j=1)
    
    p_x = dde.grad.jacobian(p, x, i=0, j=0)
    p_y = dde.grad.jacobian(p, x, i=0, j=1)
    
    momentum_x = -p_x + mu * (u_xx + u_yy)
    momentum_y = -p_y + mu * (v_xx + v_yy)
    continuity = u_x + v_y
    
    return [momentum_x, momentum_y, continuity]

# Данные
data = dde.data.PDE(
    geom, stokes_pde,
    [bc_inlet_u, bc_inlet_v, bc_wall_u, bc_wall_v, bc_out_p],
    num_domain=1000,
    num_boundary=200,
    num_test=200
)

# Сеть (уменьшил для скорости)
net = dde.nn.FNN([2] + [64] * 3 + [3], "tanh", "Glorot uniform")
model = dde.Model(data, net)

# Компиляция БЕЗ метрик
model.compile("adam", lr=0.001, loss="MSE")

# Обучение
print("Обучение PINN модели Стокса...")
losshistory, train_state = model.train(iterations=3000, display_every=500)

# Визуализация
nx, ny = 40, 20
x = np.linspace(0, L, nx)
y = np.linspace(0, H, ny)
X, Y = np.meshgrid(x, y)
points = np.hstack((X.reshape(-1,1), Y.reshape(-1,1)))

output = model.predict(points)
U = output[:, 0].reshape(nx, ny)
V = output[:, 1].reshape(nx, ny)
P = output[:, 2].reshape(nx, ny)

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

axes[0, 0].streamplot(X, Y, U.T, V.T, color='b', density=1.5)
axes[0, 0].set_title('Streamlines')
axes[0, 0].set_xlabel('x'); axes[0, 0].set_ylabel('y')
axes[0, 0].grid(True)

cf1 = axes[0, 1].contourf(X, Y, U.T, levels=20, cmap='RdBu')
plt.colorbar(cf1, ax=axes[0, 1], label='u')
axes[0, 1].set_title('Horizontal velocity u')
axes[0, 1].set_xlabel('x'); axes[0, 1].set_ylabel('y')
axes[0, 1].grid(True)

cf2 = axes[1, 0].contourf(X, Y, V.T, levels=20, cmap='RdBu')
plt.colorbar(cf2, ax=axes[1, 0], label='v')
axes[1, 0].set_title('Vertical velocity v')
axes[1, 0].set_xlabel('x'); axes[1, 0].set_ylabel('y')
axes[1, 0].grid(True)

cf3 = axes[1, 1].contourf(X, Y, P.T, levels=20, cmap='viridis')
plt.colorbar(cf3, ax=axes[1, 1], label='p')
axes[1, 1].set_title('Pressure')
axes[1, 1].set_xlabel('x'); axes[1, 1].set_ylabel('y')
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(run_dir, 'stokes_solution.png'), dpi=150)
plt.show()

# Проверка расхода
y_sec = np.linspace(0, H, 50)
print("\nMass flow rate through sections:")
flows = []
for x0 in [0.5, 1.0, 1.5]:
    pts = np.array([[x0, y] for y in y_sec])
    u_vals = model.predict(pts)[:, 0]
    flow = np.trapz(u_vals, y_sec)
    flows.append(flow)
    print(f"  x = {x0}: {flow:.4f}")

# Сохранение результатов
with open(os.path.join(run_dir, 'results.txt'), 'w') as f:
    f.write("STOKES FLOW IN A CHANNEL\n")
    f.write("="*50 + "\n")
    f.write(f"Domain: x in [0, {L}], y in [0, {H}]\n")
    f.write(f"Iterations: {len(losshistory.loss_train) * 500}\n")
    if len(losshistory.loss_train) > 0:
        if isinstance(losshistory.loss_train[-1], (list, np.ndarray)):
            final_loss = np.sum(losshistory.loss_train[-1])
        else:
            final_loss = losshistory.loss_train[-1]
        f.write(f"Final loss: {final_loss:.6f}\n")
        f.write(f"Best loss: {np.min(losshistory.loss_train):.6f}\n")
    f.write("\nMass flow rates:\n")
    for i, x0 in enumerate([0.5, 1.0, 1.5]):
        f.write(f"  x = {x0}: {flows[i]:.4f}\n")
    f.write(f"\nTheoretical flow rate: 0.3333\n")

# Профиль скорости
y_profile = np.linspace(0, H, 50)
x_mid = 1.0
pts_mid = np.array([[x_mid, y] for y in y_profile])
u_mid = model.predict(pts_mid)[:, 0]
u_theory = 4 * (y_profile / H) * (1 - y_profile / H)

plt.figure(figsize=(8, 6))
plt.plot(u_mid, y_profile, 'b-', linewidth=2, label='PINN solution')
plt.plot(u_theory, y_profile, 'r--', linewidth=2, label='Theory (Poiseuille)')
plt.xlabel('u(x=1.0, y)')
plt.ylabel('y')
plt.legend()
plt.title('Velocity profile at mid-channel')
plt.grid(True)
plt.savefig(os.path.join(run_dir, 'velocity_profile.png'), dpi=150)
plt.show()

# Вывод статистики
print("\n" + "="*50)
print("RESULTS SUMMARY")
print("="*50)
print(f"Final loss: {final_loss:.6f}")
print(f"Best loss: {np.min(losshistory.loss_train):.6f}")
print(f"\nMass conservation error: {abs(flows[0] - flows[-1])/flows[0]*100:.2f}%")
print(f"Average flow rate: {np.mean(flows):.4f} (theoretical: 0.3333)")
print("="*50)
print(f"\n✅ Results saved in: {run_dir}")