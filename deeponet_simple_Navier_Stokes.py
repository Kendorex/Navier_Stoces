import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

# ============================================
# DeepONet + PINN для 2D течения в трещине (Навье-Стокс)
# ============================================

# Создаем папку для результатов
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f"results_{timestamp}"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(f"{results_dir}/plots", exist_ok=True)
os.makedirs(f"{results_dir}/models", exist_ok=True)

print(f"Результаты будут сохранены в папку: {results_dir}")

# архитектура DeepONet
class DeepONet2D(nn.Module):
    def __init__(self, n_sensors=10, hidden_dim=40):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Linear(n_sensors, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.trunk = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, boundary_vals, coords):
        b = self.branch(boundary_vals)
        t = self.trunk(coords)
        return torch.sum(b.unsqueeze(1) * t, dim=-1)

# полная модель для скорости и давления
class NavierStokesDeepONet(nn.Module):
    def __init__(self, n_sensors=10):
        super().__init__()
        self.u_net = DeepONet2D(n_sensors, hidden_dim=40)
        self.v_net = DeepONet2D(n_sensors, hidden_dim=40)
        self.p_net = DeepONet2D(n_sensors, hidden_dim=40)
        
    def forward(self, boundary_vals, coords):
        u = self.u_net(boundary_vals, coords)
        v = self.v_net(boundary_vals, coords)
        p = self.p_net(boundary_vals, coords)
        return u, v, p


# 3. Physics-Informed Loss для Навье-Стокса
def navier_stokes_loss(model, boundary_vals, coords, nu=0.01):
    x = coords[..., 0:1].float().requires_grad_(True)
    y = coords[..., 1:2].float().requires_grad_(True)
    coords_grad = torch.cat([x, y], dim=-1)
    
    u, v, p = model(boundary_vals, coords_grad)
    
    du_dx = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    du_dy = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
    dv_dx = torch.autograd.grad(v.sum(), x, create_graph=True)[0]
    dv_dy = torch.autograd.grad(v.sum(), y, create_graph=True)[0]
    dp_dx = torch.autograd.grad(p.sum(), x, create_graph=True)[0]
    dp_dy = torch.autograd.grad(p.sum(), y, create_graph=True)[0]
    
    d2u_dx2 = torch.autograd.grad(du_dx.sum(), x, create_graph=True)[0]
    d2u_dy2 = torch.autograd.grad(du_dy.sum(), y, create_graph=True)[0]
    d2v_dx2 = torch.autograd.grad(dv_dx.sum(), x, create_graph=True)[0]
    d2v_dy2 = torch.autograd.grad(dv_dy.sum(), y, create_graph=True)[0]
    
    momentum_x = u.unsqueeze(-1)*du_dx + v.unsqueeze(-1)*du_dy + dp_dx - nu*(d2u_dx2 + d2u_dy2)
    momentum_y = u.unsqueeze(-1)*dv_dx + v.unsqueeze(-1)*dv_dy + dp_dy - nu*(d2v_dx2 + d2v_dy2)
    continuity = du_dx + dv_dy
    
    return torch.mean(momentum_x**2) + torch.mean(momentum_y**2) + torch.mean(continuity**2)


# 4. Генерация данных для течения Пуазейля в трещине
def make_poiseuille_data(n_samples=100, n_sensors=10):
    nu = 0.01
    H = 0.5
    L = 2.0
    
    y_sensors = np.linspace(0, H, n_sensors)
    
    boundary_list = []
    coords_list = []
    u_list, v_list, p_list = [], [], []
    
    for _ in range(n_samples):
        p_in = np.random.uniform(1.0, 3.0)
        p_out = np.random.uniform(0.0, 0.5)
        dp_dx = (p_out - p_in) / L
        
        boundary_vals = np.concatenate([
            p_in * np.ones(n_sensors),
            np.zeros(n_sensors)
        ])
        
        nx, ny = 30, 20
        x_grid = np.linspace(0, L, nx)
        y_grid = np.linspace(0, H, ny)
        X, Y = np.meshgrid(x_grid, y_grid)
        coords = np.stack([X.flatten(), Y.flatten()], axis=-1)
        
        u = -1/(2*nu) * dp_dx * Y * (H - Y)
        v = np.zeros_like(X)
        p = p_in + dp_dx * X
        
        boundary_list.append(boundary_vals)
        coords_list.append(coords)
        u_list.append(u.flatten())
        v_list.append(v.flatten())
        p_list.append(p.flatten())
    
    boundary_array = np.array(boundary_list)
    coords_array = np.array(coords_list)
    u_array = np.array(u_list)
    v_array = np.array(v_list)
    p_array = np.array(p_list)
    
    return (torch.FloatTensor(boundary_array),
            torch.FloatTensor(coords_array),
            torch.FloatTensor(u_array),
            torch.FloatTensor(v_array),
            torch.FloatTensor(p_array),
            L, H, nu)


def plot_training_history(losses, save_path):
    """График истории обучения"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(losses['total'], label='Total Loss', linewidth=2)
    ax.semilogy(losses['data'], label='Data Loss', alpha=0.7)
    ax.semilogy(losses['pde'], label='PDE Loss', alpha=0.7)
    ax.legend(fontsize=12)
    ax.set_title('Training History', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_results_comprehensive(X, Y, u_true_2d, u_pred_2d, v_true_2d, v_pred_2d, 
                               p_true_2d, p_pred_2d, y_grid, save_path):
    """Комплексная визуализация результатов на одном рисунке"""
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    
    # u-скорость (истинная)
    im1 = axes[0, 0].contourf(X, Y, u_true_2d, levels=20, cmap='RdBu_r')
    axes[0, 0].set_title('u-velocity (True)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # u-скорость (предсказанная)
    im2 = axes[0, 1].contourf(X, Y, u_pred_2d, levels=20, cmap='RdBu_r')
    axes[0, 1].set_title('u-velocity (Predicted)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Давление (истинное)
    im3 = axes[0, 2].contourf(X, Y, p_true_2d, levels=20, cmap='viridis')
    axes[0, 2].set_title('Pressure (True)', fontsize=12, fontweight='bold')
    axes[0, 2].set_xlabel('x')
    axes[0, 2].set_ylabel('y')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # Давление (предсказанное)
    im4 = axes[0, 3].contourf(X, Y, p_pred_2d, levels=20, cmap='viridis')
    axes[0, 3].set_title('Pressure (Predicted)', fontsize=12, fontweight='bold')
    axes[0, 3].set_xlabel('x')
    axes[0, 3].set_ylabel('y')
    plt.colorbar(im4, ax=axes[0, 3])
    
    # Профиль скорости в центре
    mid_x = u_true_2d.shape[1] // 2
    axes[1, 0].plot(u_true_2d[:, mid_x], y_grid, 'b-', label='True', linewidth=2.5)
    axes[1, 0].plot(u_pred_2d[:, mid_x], y_grid, 'r--', label='Predicted', linewidth=2.5)
    axes[1, 0].set_title('Velocity Profile at x=L/2', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('u-velocity', fontsize=11)
    axes[1, 0].set_ylabel('y', fontsize=11)
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Профиль скорости на выходе
    axes[1, 1].plot(u_true_2d[:, -1], y_grid, 'b-', label='True', linewidth=2.5)
    axes[1, 1].plot(u_pred_2d[:, -1], y_grid, 'r--', label='Predicted', linewidth=2.5)
    axes[1, 1].set_title('Velocity Profile at x=L', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('u-velocity', fontsize=11)
    axes[1, 1].set_ylabel('y', fontsize=11)
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Поле скорости (векторное)
    skip = 3
    axes[1, 2].quiver(X[::skip, ::skip], Y[::skip, ::skip], 
                   u_pred_2d[::skip, ::skip], v_pred_2d[::skip, ::skip],
                   scale=50, width=0.003)
    axes[1, 2].set_title('Predicted Velocity Field', fontsize=12, fontweight='bold')
    axes[1, 2].set_xlabel('x', fontsize=11)
    axes[1, 2].set_ylabel('y', fontsize=11)
    axes[1, 2].set_aspect('equal')
    axes[1, 2].grid(True, alpha=0.3)
    
    # Давление по центру
    axes[1, 3].plot(X[0, :], p_true_2d[0, :], 'b-', label='True', linewidth=2.5)
    axes[1, 3].plot(X[0, :], p_pred_2d[0, :], 'r--', label='Predicted', linewidth=2.5)
    axes[1, 3].set_title('Pressure along y=0', fontsize=12, fontweight='bold')
    axes[1, 3].set_xlabel('x', fontsize=11)
    axes[1, 3].set_ylabel('Pressure', fontsize=11)
    axes[1, 3].legend(fontsize=10)
    axes[1, 3].grid(True, alpha=0.3)
    
    plt.suptitle('DeepONet + PINN: Poiseuille Flow in Fracture', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# 5. Подготовка данных
print("Генерируем данные для течения в трещине...")
boundary, coords, u_true, v_true, p_true, L, H, nu = make_poiseuille_data(100)

# Разделение на train/test
train_size = 80
b_train, b_test = boundary[:train_size], boundary[train_size:]
c_train, c_test = coords[:train_size], coords[train_size:]
u_train, u_test = u_true[:train_size], u_true[train_size:]
v_train, v_test = v_true[:train_size], v_true[train_size:]
p_train, p_test = p_true[:train_size], p_true[train_size:]

print(f"Данные: {train_size} тренировочных, {100-train_size} тестовых примеров")
print(f"Размер области: L={L}, H={H}, nu={nu}")

# 6. Обучение
print("\nОбучение DeepONet + PINN для Навье-Стокса...")
model = NavierStokesDeepONet(n_sensors=20)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                        patience=100, factor=0.5)

losses = {'total': [], 'data': [], 'pde': []}
best_loss = float('inf')

for epoch in range(1000):
    model.train()
    
    # Data loss
    u_pred, v_pred, p_pred = model(b_train, c_train)
    loss_data = (torch.mean((u_pred - u_train)**2) + 
                 torch.mean((v_pred - v_train)**2) + 
                 torch.mean((p_pred - p_train)**2))
    
    # PDE loss на случайных точках
    n_pde = 100
    x_pde = torch.rand(b_train.shape[0], n_pde, 2)
    x_pde[..., 0] *= L
    x_pde[..., 1] *= H
    
    loss_pde = navier_stokes_loss(model, b_train, x_pde, nu)
    
    # Адаптивный вес для PDE loss
    pde_weight = min(0.5, 0.1 * (1 + epoch / 500))
    
    # Общая потеря
    loss = loss_data + pde_weight * loss_pde
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    losses['total'].append(loss.item())
    losses['data'].append(loss_data.item())
    losses['pde'].append(loss_pde.item())
    
    # Сохраняем лучшую модель
    if loss.item() < best_loss:
        best_loss = loss.item()
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_loss,
        }, f"{results_dir}/models/best_model.pth")
    
    scheduler.step(loss)
    
    if epoch % 100 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:4d} | Total: {loss.item():.4f} | Data: {loss_data.item():.4f} | "
              f"PDE: {loss_pde.item():.4f} | PDE weight: {pde_weight:.3f} | LR: {current_lr:.6f}")

print("Обучение завершено!")

# Загружаем лучшую модель для тестирования
checkpoint = torch.load(f"{results_dir}/models/best_model.pth")
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Загружена лучшая модель (эпоха {checkpoint['epoch']}, loss: {checkpoint['loss']:.4f})")

# Сохраняем финальную модель
torch.save({
    'model_state_dict': model.state_dict(),
    'model_architecture': str(model),
    'hyperparameters': {
        'n_sensors': 20,
        'hidden_dim': 40,
        'L': L,
        'H': H,
        'nu': nu
    }
}, f"{results_dir}/models/final_model.pth")

# 7. Визуализация
print("\nСоздание графиков...")

# График обучения
plot_training_history(losses, f"{results_dir}/plots/training_history.png")

# Выбираем лучший тестовый пример (с минимальной ошибкой)
best_test_idx = 0
best_test_error = float('inf')

with torch.no_grad():
    for test_idx in range(len(b_test)):
        u_pred, v_pred, p_pred = model(b_test[test_idx:test_idx+1], c_test[test_idx:test_idx+1])
        error = torch.mean((u_pred - u_test[test_idx:test_idx+1])**2).item()
        if error < best_test_error:
            best_test_error = error
            best_test_idx = test_idx

print(f"Лучший тестовый пример: {best_test_idx + 1} (MSE u: {best_test_error:.6f})")

# Визуализация лучшего примера
with torch.no_grad():
    u_pred, v_pred, p_pred = model(b_test[best_test_idx:best_test_idx+1], c_test[best_test_idx:best_test_idx+1])

# Преобразуем в 2D для визуализации
nx, ny = 30, 20
u_pred_2d = u_pred.reshape(ny, nx).numpy()
v_pred_2d = v_pred.reshape(ny, nx).numpy()
p_pred_2d = p_pred.reshape(ny, nx).numpy()

u_true_2d = u_test[best_test_idx].reshape(ny, nx).numpy()
v_true_2d = v_test[best_test_idx].reshape(ny, nx).numpy()
p_true_2d = p_test[best_test_idx].reshape(ny, nx).numpy()

x_grid = np.linspace(0, L, nx)
y_grid = np.linspace(0, H, ny)
X, Y = np.meshgrid(x_grid, y_grid)

# Комплексная визуализация
plot_results_comprehensive(X, Y, u_true_2d, u_pred_2d, v_true_2d, v_pred_2d,
                           p_true_2d, p_pred_2d, y_grid, 
                           f"{results_dir}/plots/results.png")

# 8. Статистика и сохранение результатов
with torch.no_grad():
    u_test_pred, v_test_pred, p_test_pred = model(b_test, c_test)
    
    # Характерная скорость (максимальная скорость потока)
    char_vel = torch.max(torch.abs(u_test))
    
    # Относительные ошибки
    rel_error_u = torch.norm(u_test_pred - u_test) / torch.norm(u_test)
    rel_error_v = torch.norm(v_test_pred - v_test) / char_vel
    rel_error_p = torch.norm(p_test_pred - p_test) / torch.norm(p_test)
    
    # Абсолютные ошибки
    mse_u = torch.mean((u_test_pred - u_test)**2)
    mse_v = torch.mean((v_test_pred - v_test)**2)
    mse_p = torch.mean((p_test_pred - p_test)**2)
    
    # Максимальные ошибки
    max_error_u = torch.max(torch.abs(u_test_pred - u_test))
    max_error_v = torch.max(torch.abs(v_test_pred - v_test))
    max_error_p = torch.max(torch.abs(p_test_pred - p_test))

# Вывод результатов
results_text = f"""
{'='*60}
РЕЗУЛЬТАТЫ ОБУЧЕНИЯ DeepONet + PINN
{'='*60}

Финальные значения Loss:
   * Total Loss: {losses['total'][-1]:.4f}
   * Data Loss: {losses['data'][-1]:.4f}
   * PDE Loss: {losses['pde'][-1]:.4f}

Относительные ошибки:
   * u-velocity: {rel_error_u:.4f} ({rel_error_u*100:.2f}%)
   * v-velocity: {rel_error_v:.4f} ({rel_error_v*100:.2f}% от u_max)
   * pressure:   {rel_error_p:.4f} ({rel_error_p*100:.2f}%)

Среднеквадратичные ошибки (MSE):
   * u-velocity: {mse_u:.6f}
   * v-velocity: {mse_v:.6f}
   * pressure:   {mse_p:.6f}

Максимальные абсолютные ошибки:
   * u-velocity: {max_error_u:.4f}
   * v-velocity: {max_error_v:.4f}
   * pressure:   {max_error_p:.4f}

Параметры задачи:
   * Длина трещины L = {L}
   * Ширина трещины H = {H}
   * Вязкость nu = {nu}
   * Характерная скорость u_max = {char_vel:.4f}
   * Число сенсоров = 20
   * Размер скрытого слоя = 40

Сохраненные файлы:
   * Модель: {results_dir}/models/
   * Графики: {results_dir}/plots/
"""

print(results_text)

# Сохраняем результаты в текстовый файл
with open(f"{results_dir}/results_summary.txt", 'w', encoding='utf-8') as f:
    f.write(results_text)

print(f"Все результаты сохранены в папку: {results_dir}/")
print(f"   {results_dir}/plots/ - графики")
print(f"   {results_dir}/models/ - модели")
print(f"   {results_dir}/results_summary.txt - сводка результатов")