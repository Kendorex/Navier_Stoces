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

# ========== gPINN: PDE + градиенты PDE ==========
def stokes_pde_with_gradients(x, y):
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
    
    # Невязки PDE
    momentum_x = -p_x + mu * (u_xx + u_yy)
    momentum_y = -p_y + mu * (v_xx + v_yy)
    continuity = u_x + v_y
    
    # Градиенты невязок
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
    num_domain=5000,      
    num_boundary=800,    
    num_test=600
)

layer_size = [2] + [64, 64, 64] + [3] 
net = dde.nn.FNN(layer_size, "tanh", "Glorot uniform")

model = dde.Model(data, net)
model.compile("adam", lr=0.001, loss="MSE")
print("Обучение gPINN модели Стокса для узкого канала (L={}, H={})...".format(L, H))
print(f"Соотношение сторон: {L/H:.1f}:1")
losshistory, train_state = model.train(iterations=3000, display_every=500)


# Визуализация
nx, ny = 50, 30  # больше точек для узкого канала
x = np.linspace(0, L, nx)
y = np.linspace(0, H, ny)
X, Y = np.meshgrid(x, y)
points = np.hstack((X.reshape(-1,1), Y.reshape(-1,1)))

output = model.predict(points)
U = output[:, 0].reshape(nx, ny)
V = output[:, 1].reshape(nx, ny)
P = output[:, 2].reshape(nx, ny)

fig, axes = plt.subplots(1, 2, figsize=(8, 6))

# Линии тока
axes[0].streamplot(X, Y, U.T, V.T, color='steelblue', density=2, linewidth=1.5)
axes[0].set_title('Линии тока', fontsize=12, fontweight='bold')
axes[0].set_xlabel('x'); axes[0].set_ylabel('y')
axes[0].set_xlim(0, L)
axes[0].set_ylim(0, H)
axes[0].grid(True, alpha=0.3)
axes[0].set_aspect('auto')

# Горизонтальная скорость u (исправлено)
cf1 = axes[1].contourf(X, Y, U.T, levels=30, cmap='RdBu_r', extend='both')
cs1 = axes[1].contour(X, Y, U.T, levels=10, colors='black', linewidths=0.5, alpha=0.3)
plt.colorbar(cf1, ax=axes[1], label='u (м/с)', pad=0.05)
axes[1].set_title('Горизонтальная скорость u', fontsize=12, fontweight='bold')
axes[1].set_xlabel('x'); axes[1].set_ylabel('y')
axes[1].set_xlim(0, L)
axes[1].set_ylim(0, H)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(run_dir, 'gpinn_stokes_solution.png'), dpi=150, bbox_inches='tight')
plt.show()

# Проверка расхода
y_sec = np.linspace(0, H, 50)
print("\nMass flow rate through sections (gPINN):")
flows = []
# Для узкого канала проверяем в трёх сечениях
for x0 in [0.25, 0.5, 0.75]:
    pts = np.array([[x0, y] for y in y_sec])
    u_vals = model.predict(pts)[:, 0]
    flow = np.trapz(u_vals, y_sec)
    flows.append(flow)
    print(f"  x = {x0}: {flow:.4f}")

# Теоретический расход для параболического профиля
# ∫_0^H 4*(y/H)*(1-y/H) dy = 2H/3 = 0.06667
theoretical_flow = 2 * H / 3
print(f"\nTheoretical flow rate: {theoretical_flow:.4f}")

# Профиль скорости в середине канала
y_profile = np.linspace(0, H, 50)
x_mid = 0.5
pts_mid = np.array([[x_mid, y] for y in y_profile])
u_mid = model.predict(pts_mid)[:, 0]
u_theory = 4 * (y_profile / H) * (1 - y_profile / H)

plt.figure(figsize=(8, 6))
plt.plot(u_mid, y_profile, 'b-', linewidth=2, label='gPINN solution')
plt.plot(u_theory, y_profile, 'r--', linewidth=2, label='Theory (Poiseuille)')
plt.xlabel('u(x=0.5, y)')
plt.ylabel('y')
plt.legend()
plt.title('Velocity profile at mid-channel - gPINN (narrow channel)')
plt.grid(True)
plt.savefig(os.path.join(run_dir, 'velocity_profile.png'), dpi=150)
plt.show()

# Сохранение результатов
print("\n" + "="*50)
print("gPINN RESULTS SUMMARY (Narrow Channel)")
print("="*50)
print(f"Channel dimensions: L = {L}, H = {H} (aspect ratio = {L/H:.0f}:1)")

if isinstance(losshistory.loss_train[-1], np.ndarray):
    final_loss_total = np.sum(losshistory.loss_train[-1])
    print(f"Final loss (total sum): {final_loss_total:.6f}")
    print(f"PDE components loss: {np.sum(losshistory.loss_train[-1][:3]):.6f}")
    print(f"Gradient components loss: {np.sum(losshistory.loss_train[-1][3:9]):.6f}")
    print(f"BC components loss: {np.sum(losshistory.loss_train[-1][9:]):.6f}")
else:
    print(f"Final loss: {losshistory.loss_train[-1]:.6f}")

if isinstance(losshistory.loss_train, list):
    if len(losshistory.loss_train) > 0:
        if isinstance(losshistory.loss_train[0], np.ndarray):
            best_loss_total = min(np.sum(loss) for loss in losshistory.loss_train)
            print(f"Best loss (total sum): {best_loss_total:.6f}")

print(f"\nMass conservation error: {abs(flows[0] - flows[-1])/flows[0]*100:.2f}%")
print(f"Average flow rate: {np.mean(flows):.4f} (theoretical: {theoretical_flow:.4f})")
print(f"Error vs theory: {abs(np.mean(flows) - theoretical_flow)/theoretical_flow*100:.2f}%")
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

# ========== gPINN: PDE + градиенты PDE ==========
def stokes_pde_with_gradients(x, y):
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
    
    # Невязки PDE
    momentum_x = -p_x + mu * (u_xx + u_yy)
    momentum_y = -p_y + mu * (v_xx + v_yy)
    continuity = u_x + v_y
    
    # Градиенты невязок
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
    num_domain=2000,      
    num_boundary=400,    
    num_test=400
)

layer_size = [2] + [64, 64, 64] + [3] 
net = dde.nn.FNN(layer_size, "tanh", "Glorot uniform")

model = dde.Model(data, net)
model.compile("adam", lr=0.001, loss="MSE")
print("Обучение gPINN модели Стокса для узкого канала (L={}, H={})...".format(L, H))
print(f"Соотношение сторон: {L/H:.1f}:1")
losshistory, train_state = model.train(iterations=3000, display_every=500)


# Визуализация
nx, ny = 50, 30  # больше точек для узкого канала
x = np.linspace(0, L, nx)
y = np.linspace(0, H, ny)
X, Y = np.meshgrid(x, y)
points = np.hstack((X.reshape(-1,1), Y.reshape(-1,1)))

output = model.predict(points)
U = output[:, 0].reshape(nx, ny)
V = output[:, 1].reshape(nx, ny)
P = output[:, 2].reshape(nx, ny)

fig, axes = plt.subplots(1, 2, figsize=(8, 6))

# Линии тока
axes[0].streamplot(X, Y, U.T, V.T, color='steelblue', density=2, linewidth=1.5)
axes[0].set_title('Линии тока', fontsize=12, fontweight='bold')
axes[0].set_xlabel('x'); axes[0].set_ylabel('y')
axes[0].set_xlim(0, L)
axes[0].set_ylim(0, H)
axes[0].grid(True, alpha=0.3)
axes[0].set_aspect('auto')

# Горизонтальная скорость u (исправлено)
cf1 = axes[1].contourf(X, Y, U.T, levels=30, cmap='RdBu_r', extend='both')
cs1 = axes[1].contour(X, Y, U.T, levels=10, colors='black', linewidths=0.5, alpha=0.3)
plt.colorbar(cf1, ax=axes[1], label='u (м/с)', pad=0.05)
axes[1].set_title('Горизонтальная скорость u', fontsize=12, fontweight='bold')
axes[1].set_xlabel('x'); axes[1].set_ylabel('y')
axes[1].set_xlim(0, L)
axes[1].set_ylim(0, H)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(run_dir, 'gpinn_stokes_solution.png'), dpi=150, bbox_inches='tight')
plt.show()

# Проверка расхода
y_sec = np.linspace(0, H, 50)
print("\nMass flow rate through sections (gPINN):")
flows = []
# Для узкого канала проверяем в трёх сечениях
for x0 in [0.25, 0.5, 0.75]:
    pts = np.array([[x0, y] for y in y_sec])
    u_vals = model.predict(pts)[:, 0]
    flow = np.trapz(u_vals, y_sec)
    flows.append(flow)
    print(f"  x = {x0}: {flow:.4f}")

# Теоретический расход для параболического профиля
# ∫_0^H 4*(y/H)*(1-y/H) dy = 2H/3 = 0.06667
theoretical_flow = 2 * H / 3
print(f"\nTheoretical flow rate: {theoretical_flow:.4f}")

# Профиль скорости в середине канала
y_profile = np.linspace(0, H, 50)
x_mid = 0.5
pts_mid = np.array([[x_mid, y] for y in y_profile])
u_mid = model.predict(pts_mid)[:, 0]
u_theory = 4 * (y_profile / H) * (1 - y_profile / H)

plt.figure(figsize=(8, 6))
plt.plot(u_mid, y_profile, 'b-', linewidth=2, label='gPINN solution')
plt.plot(u_theory, y_profile, 'r--', linewidth=2, label='Theory (Poiseuille)')
plt.xlabel('u(x=0.5, y)')
plt.ylabel('y')
plt.legend()
plt.title('Velocity profile at mid-channel - gPINN (narrow channel)')
plt.grid(True)
plt.savefig(os.path.join(run_dir, 'velocity_profile.png'), dpi=150)
plt.show()

# Сохранение результатов
print("\n" + "="*50)
print("gPINN RESULTS SUMMARY (Narrow Channel)")
print("="*50)
print(f"Channel dimensions: L = {L}, H = {H} (aspect ratio = {L/H:.0f}:1)")

if isinstance(losshistory.loss_train[-1], np.ndarray):
    final_loss_total = np.sum(losshistory.loss_train[-1])
    print(f"Final loss (total sum): {final_loss_total:.6f}")
    print(f"PDE components loss: {np.sum(losshistory.loss_train[-1][:3]):.6f}")
    print(f"Gradient components loss: {np.sum(losshistory.loss_train[-1][3:9]):.6f}")
    print(f"BC components loss: {np.sum(losshistory.loss_train[-1][9:]):.6f}")
else:
    print(f"Final loss: {losshistory.loss_train[-1]:.6f}")

if isinstance(losshistory.loss_train, list):
    if len(losshistory.loss_train) > 0:
        if isinstance(losshistory.loss_train[0], np.ndarray):
            best_loss_total = min(np.sum(loss) for loss in losshistory.loss_train)
            print(f"Best loss (total sum): {best_loss_total:.6f}")

print(f"\nMass conservation error: {abs(flows[0] - flows[-1])/flows[0]*100:.2f}%")
print(f"Average flow rate: {np.mean(flows):.4f} (theoretical: {theoretical_flow:.4f})")
print(f"Error vs theory: {abs(np.mean(flows) - theoretical_flow)/theoretical_flow*100:.2f}%")
