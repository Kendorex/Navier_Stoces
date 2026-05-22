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

bc_inlet_u = dde.icbc.DirichletBC(geom, inlet_u, inflow, component=0)
bc_inlet_v = dde.icbc.DirichletBC(geom, lambda x: 0.0, inflow, component=1)
bc_wall_u = dde.icbc.DirichletBC(geom, lambda x: 0.0, wall, component=0)
bc_wall_v = dde.icbc.DirichletBC(geom, lambda x: 0.0, wall, component=1)
bc_out_p = dde.icbc.DirichletBC(geom, lambda x: 0.0, outflow, component=2)

def stokes_pde_with_gradients(x, y):
    u = tf.cast(y[:, 0:1], tf.float32)
    v = tf.cast(y[:, 1:2], tf.float32)
    p = tf.cast(y[:, 2:3], tf.float32)
    mu = tf.constant(1.0, dtype=tf.float32)
    
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
    
    momentum_x = -p_x + mu * (u_xx + u_yy)
    momentum_y = -p_y + mu * (v_xx + v_yy)
    continuity = u_x + v_y
    
    g_mom_x_x = tf.cast(dde.grad.jacobian(momentum_x, x, i=0, j=0), tf.float32)
    g_mom_x_y = tf.cast(dde.grad.jacobian(momentum_x, x, i=0, j=1), tf.float32)
    g_mom_y_x = tf.cast(dde.grad.jacobian(momentum_y, x, i=0, j=0), tf.float32)
    g_mom_y_y = tf.cast(dde.grad.jacobian(momentum_y, x, i=0, j=1), tf.float32)
    g_cont_x = tf.cast(dde.grad.jacobian(continuity, x, i=0, j=0), tf.float32)
    g_cont_y = tf.cast(dde.grad.jacobian(continuity, x, i=0, j=1), tf.float32)
    
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
print("Обучение gPINN модели Стокса (L={}, H={})...".format(L, H))
losshistory, train_state = model.train(iterations=3000, display_every=500)


def plot_training_loss(losshistory, filename):
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


def plot_results_comprehensive(filename):
    GRID_NX, GRID_NY = 200, 80
    xs = np.linspace(0, L, GRID_NX, dtype=np.float32)
    ys = np.linspace(0, H, GRID_NY, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)
    pts = np.column_stack([X.ravel(), Y.ravel()]).astype(np.float32)
    
    output = model.predict(pts)
    u = output[:, 0].reshape(GRID_NY, GRID_NX)
    v = output[:, 1].reshape(GRID_NY, GRID_NX)
    p = output[:, 2].reshape(GRID_NY, GRID_NX)
    speed = np.sqrt(u**2 + v**2)
    
    def poiseuille_profile(y, h=H):
        return 4.0 * (y / h) * (1.0 - y / h)
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    
    x_wall = [0, L, L, 0, 0]
    y_wall_top = [H, H, H, H, H]
    y_wall_bottom = [0, 0, 0, 0, 0]
    
    # u-скорость
    im1 = axes[0, 0].contourf(X, Y, u, levels=30, cmap='RdBu_r')
    axes[0, 0].plot(x_wall, y_wall_top, 'k-', linewidth=1.5)
    axes[0, 0].plot(x_wall, y_wall_bottom, 'k-', linewidth=1.5)
    axes[0, 0].set_title('u-velocity', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    axes[0, 0].set_aspect('equal')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Speed
    im2 = axes[0, 1].contourf(X, Y, speed, levels=30, cmap='plasma')
    axes[0, 1].plot(x_wall, y_wall_top, 'k-', linewidth=1.5)
    axes[0, 1].plot(x_wall, y_wall_bottom, 'k-', linewidth=1.5)
    axes[0, 1].set_title('Speed', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    axes[0, 1].set_aspect('equal')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Давление
    im3 = axes[0, 2].contourf(X, Y, p, levels=30, cmap='viridis')
    axes[0, 2].plot(x_wall, y_wall_top, 'k-', linewidth=1.5)
    axes[0, 2].plot(x_wall, y_wall_bottom, 'k-', linewidth=1.5)
    axes[0, 2].set_title('Pressure', fontsize=12, fontweight='bold')
    axes[0, 2].set_xlabel('x')
    axes[0, 2].set_ylabel('y')
    axes[0, 2].set_aspect('equal')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # v-скорость
    im4 = axes[0, 3].contourf(X, Y, v, levels=30, cmap='RdBu_r')
    axes[0, 3].plot(x_wall, y_wall_top, 'k-', linewidth=1.5)
    axes[0, 3].plot(x_wall, y_wall_bottom, 'k-', linewidth=1.5)
    axes[0, 3].set_title('v-velocity', fontsize=12, fontweight='bold')
    axes[0, 3].set_xlabel('x')
    axes[0, 3].set_ylabel('y')
    axes[0, 3].set_aspect('equal')
    plt.colorbar(im4, ax=axes[0, 3])
    
    # Профиль скорости на входе (x=0.2)
    inlet_idx = np.argmin(np.abs(xs - 0.2))
    y_plot = ys
    u_inlet = u[:, inlet_idx]
    
    axes[1, 0].plot(u_inlet, y_plot, 'b-', label='gPINN', linewidth=2.5)
    u_theory = poiseuille_profile(y_plot)
    axes[1, 0].plot(u_theory, y_plot, 'r--', label='Poiseuille', linewidth=2)
    axes[1, 0].set_title('Velocity Profile at Inlet (x=0.2)', fontsize=11, fontweight='bold')
    axes[1, 0].set_xlabel('u-velocity', fontsize=10)
    axes[1, 0].set_ylabel('y', fontsize=10)
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Профиль скорости в середине (x=1.0)
    mid_idx = np.argmin(np.abs(xs - 1.0))
    u_mid = u[:, mid_idx]
    
    axes[1, 1].plot(u_mid, y_plot, 'b-', label='gPINN', linewidth=2.5)
    u_theory = poiseuille_profile(y_plot)
    axes[1, 1].plot(u_theory, y_plot, 'r--', label='Poiseuille', linewidth=2)
    axes[1, 1].set_title('Velocity Profile at Mid (x=1.0)', fontsize=11, fontweight='bold')
    axes[1, 1].set_xlabel('u-velocity', fontsize=10)
    axes[1, 1].set_ylabel('y', fontsize=10)
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Поле скорости
    step = 8
    X_sub = X[::step, ::step]
    Y_sub = Y[::step, ::step]
    u_sub = u[::step, ::step]
    v_sub = v[::step, ::step]
    
    axes[1, 2].plot(x_wall, y_wall_top, 'k-', linewidth=1.5)
    axes[1, 2].plot(x_wall, y_wall_bottom, 'k-', linewidth=1.5)
    axes[1, 2].quiver(X_sub, Y_sub, u_sub, v_sub, scale=15, width=0.003)
    axes[1, 2].set_title('Velocity Field', fontsize=12, fontweight='bold')
    axes[1, 2].set_xlabel('x', fontsize=11)
    axes[1, 2].set_ylabel('y', fontsize=11)
    axes[1, 2].set_aspect('equal')
    axes[1, 2].grid(True, alpha=0.3)
    
    # Давление по центру
    center_idx = GRID_NY // 2
    p_center = p[center_idx, :]
    
    axes[1, 3].plot(xs, p_center, 'b-', linewidth=2.5)
    axes[1, 3].set_title('Pressure along Centerline', fontsize=12, fontweight='bold')
    axes[1, 3].set_xlabel('x', fontsize=11)
    axes[1, 3].set_ylabel('Pressure', fontsize=11)
    axes[1, 3].grid(True, alpha=0.3)
    
    plt.suptitle('gPINN: Stokes Flow in Channel', fontsize=14, fontweight='bold')
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

print("\n" + "="*50)
print("gPINN RESULTS SUMMARY")
print("="*50)
print(f"Channel: L={L}, H={H} (aspect ratio={L/H:.0f}:1)")

if isinstance(losshistory.loss_train[-1], np.ndarray):
    final_loss_total = np.sum(losshistory.loss_train[-1])
    print(f"Final loss: {final_loss_total:.6f}")
else:
    print(f"Final loss: {losshistory.loss_train[-1]:.6f}")

print(f"Avg flow rate: {np.mean(flows):.4f} (theoretical: {theoretical_flow:.4f})")
print(f"Error: {abs(np.mean(flows)-theoretical_flow)/theoretical_flow*100:.2f}%")
print(f"\nВсе результаты сохранены в: {run_dir}")