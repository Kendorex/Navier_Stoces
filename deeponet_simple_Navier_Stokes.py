import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# ============================================
# DeepONet + PINN для 2D течения в трещине (Навье-Стокс)
# ============================================

# 1. Архитектура DeepONet для 2D
class DeepONet2D(nn.Module):
    def __init__(self, n_sensors=10, hidden_dim=40):
        super().__init__()
        # Branch: обрабатывает граничные условия (давление на входе/выходе)
        self.branch = nn.Sequential(
            nn.Linear(n_sensors, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Trunk: обрабатывает координаты (x, y)
        self.trunk = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, boundary_vals, coords):
        # boundary_vals: (batch, n_sensors) - значения на границе
        # coords: (batch, N, 2) - координаты (x, y)
        b = self.branch(boundary_vals)  # (batch, hidden_dim)
        t = self.trunk(coords)          # (batch, N, hidden_dim)
        return torch.sum(b.unsqueeze(1) * t, dim=-1)  # (batch, N)


# 2. Полная модель для скорости и давления
class NavierStokesDeepONet(nn.Module):
    def __init__(self, n_sensors=10):
        super().__init__()
        # Отдельные DeepONet для u, v, p
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
    """
    Уравнения Навье-Стокса для несжимаемой жидкости:
    1. u·∇u + ∇p - ν∇²u = 0  (сохранение импульса)
    2. ∇·u = 0                 (несжимаемость)
    """
    x = coords[..., 0:1].float().requires_grad_(True)
    y = coords[..., 1:2].float().requires_grad_(True)
    coords_grad = torch.cat([x, y], dim=-1)
    
    u, v, p = model(boundary_vals, coords_grad)
    
    # Первые производные
    du_dx = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    du_dy = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
    dv_dx = torch.autograd.grad(v.sum(), x, create_graph=True)[0]
    dv_dy = torch.autograd.grad(v.sum(), y, create_graph=True)[0]
    dp_dx = torch.autograd.grad(p.sum(), x, create_graph=True)[0]
    dp_dy = torch.autograd.grad(p.sum(), y, create_graph=True)[0]
    
    # Вторые производные
    d2u_dx2 = torch.autograd.grad(du_dx.sum(), x, create_graph=True)[0]
    d2u_dy2 = torch.autograd.grad(du_dy.sum(), y, create_graph=True)[0]
    d2v_dx2 = torch.autograd.grad(dv_dx.sum(), x, create_graph=True)[0]
    d2v_dy2 = torch.autograd.grad(dv_dy.sum(), y, create_graph=True)[0]
    
    # Уравнение импульса по x: u*du/dx + v*du/dy + dp/dx - nu*(d2u/dx2 + d2u/dy2) = 0
    momentum_x = u.unsqueeze(-1)*du_dx + v.unsqueeze(-1)*du_dy + dp_dx - nu*(d2u_dx2 + d2u_dy2)
    
    # Уравнение импульса по y: u*dv/dx + v*dv/dy + dp/dy - nu*(d2v/dx2 + d2v/dy2) = 0
    momentum_y = u.unsqueeze(-1)*dv_dx + v.unsqueeze(-1)*dv_dy + dp_dy - nu*(d2v_dx2 + d2v_dy2)
    
    # Уравнение неразрывности: du/dx + dv/dy = 0
    continuity = du_dx + dv_dy
    
    return torch.mean(momentum_x**2) + torch.mean(momentum_y**2) + torch.mean(continuity**2)


# 4. Генерация данных для течения Пуазейля в трещине
def make_poiseuille_data(n_samples=100, n_sensors=10):
    """
    Течение Пуазейля: параболический профиль скорости
    u(y) = -1/(2*nu) * dp/dx * y*(H-y)
    v = 0
    p(x) = p_in - (p_in - p_out)*x/L
    """
    nu = 0.01  # вязкость
    H = 1.0    # ширина трещины
    L = 2.0    # длина трещины
    
    # Сенсоры на входе (измеряем давление)
    y_sensors = np.linspace(0, H, n_sensors)
    
    boundary_list = []
    coords_list = []
    u_list, v_list, p_list = [], [], []
    
    for _ in range(n_samples):
        # Случайный перепад давления
        p_in = np.random.uniform(1.0, 3.0)
        p_out = np.random.uniform(0.0, 0.5)
        dp_dx = (p_out - p_in) / L
        
        # Значения на входной границе (давление + скорость)
        boundary_vals = np.concatenate([
            p_in * np.ones(n_sensors),  # давление на входе
            np.zeros(n_sensors)          # v=0 на входе
        ])
        
        # Создаём сетку точек внутри области
        nx, ny = 30, 20
        x_grid = np.linspace(0, L, nx)
        y_grid = np.linspace(0, H, ny)
        X, Y = np.meshgrid(x_grid, y_grid)
        coords = np.stack([X.flatten(), Y.flatten()], axis=-1)
        
        # Аналитическое решение
        u = -1/(2*nu) * dp_dx * Y * (H - Y)
        v = np.zeros_like(X)
        p = p_in + dp_dx * X
        
        boundary_list.append(boundary_vals)
        coords_list.append(coords)
        u_list.append(u.flatten())
        v_list.append(v.flatten())
        p_list.append(p.flatten())
    
    return (torch.FloatTensor(boundary_list),
            torch.FloatTensor(coords_list),
            torch.FloatTensor(u_list),
            torch.FloatTensor(v_list),
            torch.FloatTensor(p_list),
            L, H, nu)

# 5. Подготовка данных
print("📊 Генерируем данные для течения в трещине...")
boundary, coords, u_true, v_true, p_true, L, H, nu = make_poiseuille_data(100)

# Разделение на train/test
train_size = 80
b_train, b_test = boundary[:train_size], boundary[train_size:]
c_train, c_test = coords[:train_size], coords[train_size:]
u_train, u_test = u_true[:train_size], u_true[train_size:]
v_train, v_test = v_true[:train_size], v_true[train_size:]
p_train, p_test = p_true[:train_size], p_true[train_size:]

# 6. Обучение
print("🚀 Обучение DeepONet + PINN для Навье-Стокса...")
model = NavierStokesDeepONet(n_sensors=20)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

losses = {'total': [], 'data': [], 'pde': []}

for epoch in range(1000):
    # Data loss
    u_pred, v_pred, p_pred = model(b_train, c_train)
    loss_data = (torch.mean((u_pred - u_train)**2) + 
                 torch.mean((v_pred - v_train)**2) + 
                 torch.mean((p_pred - p_train)**2))
    
    # PDE loss на случайных точках
    n_pde = 100
    x_pde = torch.rand(b_train.shape[0], n_pde, 2)
    x_pde[..., 0] *= L  # масштабируем x
    x_pde[..., 1] *= H  # масштабируем y
    
    loss_pde = navier_stokes_loss(model, b_train, x_pde, nu)
    
    # Общая потеря
    loss = loss_data + 0.1 * loss_pde
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses['total'].append(loss.item())
    losses['data'].append(loss_data.item())
    losses['pde'].append(loss_pde.item())
    
    if epoch % 200 == 0:
        print(f"Epoch {epoch:4d} | Total: {loss.item():.4f} | Data: {loss_data.item():.4f} | PDE: {loss_pde.item():.4f}")

print("✅ Обучение завершено!")

# 7. Визуализация
fig, axes = plt.subplots(2, 4, figsize=(18, 8))

# График потерь
axes[0, 0].semilogy(losses['total'], label='Total')
axes[0, 0].semilogy(losses['data'], label='Data', alpha=0.7)
axes[0, 0].semilogy(losses['pde'], label='PDE', alpha=0.7)
axes[0, 0].legend()
axes[0, 0].set_title('Training Loss')
axes[0, 0].grid(True)
axes[0, 0].set_xlabel('Epoch')

# Тестируем на новом примере
test_idx = 0
with torch.no_grad():
    u_pred, v_pred, p_pred = model(b_test[test_idx:test_idx+1], c_test[test_idx:test_idx+1])

# Преобразуем в 2D для визуализации
nx, ny = 30, 20
u_pred_2d = u_pred.reshape(ny, nx).numpy()
v_pred_2d = v_pred.reshape(ny, nx).numpy()
p_pred_2d = p_pred.reshape(ny, nx).numpy()

u_true_2d = u_test[test_idx].reshape(ny, nx).numpy()
v_true_2d = v_test[test_idx].reshape(ny, nx).numpy()
p_true_2d = p_test[test_idx].reshape(ny, nx).numpy()

x_grid = np.linspace(0, L, nx)
y_grid = np.linspace(0, H, ny)
X, Y = np.meshgrid(x_grid, y_grid)

# Визуализация u-скорости
im1 = axes[1, 0].contourf(X, Y, u_true_2d, levels=20, cmap='RdBu_r')
axes[1, 0].set_title('u (True)')
plt.colorbar(im1, ax=axes[1, 0])
axes[1, 0].set_xlabel('x')
axes[1, 0].set_ylabel('y')

im2 = axes[1, 1].contourf(X, Y, u_pred_2d, levels=20, cmap='RdBu_r')
axes[1, 1].set_title('u (Predicted)')
plt.colorbar(im2, ax=axes[1, 1])
axes[1, 1].set_xlabel('x')
axes[1, 1].set_ylabel('y')

# Профиль скорости по центру
mid_x = nx // 2
axes[0, 1].plot(u_true_2d[:, mid_x], y_grid, 'b-', label='True', linewidth=2)
axes[0, 1].plot(u_pred_2d[:, mid_x], y_grid, 'r--', label='Predicted', linewidth=2)
axes[0, 1].set_title('Velocity Profile at x=L/2')
axes[0, 1].set_xlabel('u')
axes[0, 1].set_ylabel('y')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Давление
im3 = axes[0, 2].contourf(X, Y, p_true_2d, levels=20, cmap='viridis')
axes[0, 2].set_title('Pressure (True)')
plt.colorbar(im3, ax=axes[0, 2])

im4 = axes[0, 3].contourf(X, Y, p_pred_2d, levels=20, cmap='viridis')
axes[0, 3].set_title('Pressure (Predicted)')
plt.colorbar(im4, ax=axes[0, 3])

# Ошибка u
error_u = np.abs(u_pred_2d - u_true_2d)
im5 = axes[1, 2].contourf(X, Y, error_u, levels=20, cmap='Reds')
axes[1, 2].set_title('|u Error|')
plt.colorbar(im5, ax=axes[1, 2])

# Поле скорости (стрелки)
skip = 3
axes[1, 3].quiver(X[::skip, ::skip], Y[::skip, ::skip], 
                  u_pred_2d[::skip, ::skip], v_pred_2d[::skip, ::skip])
axes[1, 3].set_title('Velocity Field')
axes[1, 3].set_xlabel('x')
axes[1, 3].set_ylabel('y')
axes[1, 3].set_aspect('equal')

plt.suptitle('DeepONet + PINN: Течение Пуазейля в трещине (Навье-Стокс)', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Статистика
with torch.no_grad():
    u_test_pred, v_test_pred, p_test_pred = model(b_test, c_test)
    rel_error_u = torch.norm(u_test_pred - u_test) / torch.norm(u_test)
    rel_error_v = torch.norm(v_test_pred - v_test) / torch.norm(v_test)
    rel_error_p = torch.norm(p_test_pred - p_test) / torch.norm(p_test)
    
print("\n" + "="*60)
print("📊 Результаты для течения в трещине:")
print("="*60)
print(f"Относительная ошибка u: {rel_error_u:.4f}")
print(f"Относительная ошибка v: {rel_error_v:.4f}")
print(f"Относительная ошибка p: {rel_error_p:.4f}")
print("\n🎯 Физика задачи:")
print("• Параболический профиль скорости (течение Пуазейля)")
print("• Удовлетворяет уравнениям Навье-Стокса")
print("• DeepONet учится для разных перепадов давления")
print("• PINN-составляющая обеспечивает выполнение законов физики")