import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os

# ============================================
# DeepONet + PINN для 2D течения в сужающейся трещине
# ============================================

# Параметры геометрии с сужением
L = 2.0
H_IN = 0.50
H_THROAT = 0.25
X_CONTRACT_1 = 0.75
X_CONTRACT_2 = 0.90
X_EXPAND_1 = 1.10
X_EXPAND_2 = 1.25

# Физические параметры
RHO = 1.0
NU = 0.02
U_MAX = 1.0

# Директория для сохранения
OUT_DIR = "deeponet_constricted_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================
# 1. Геометрия сужающейся трещины
# ============================================
def channel_width(x):
    """Полуширина канала в точке x"""
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = np.asarray(x, dtype=np.float64)
    
    w = np.zeros_like(x_np, dtype=np.float64)
    
    for i, xi in enumerate(x_np.flat):
        if xi <= X_CONTRACT_1:
            w.flat[i] = H_IN / 2
        elif xi <= X_CONTRACT_2:
            t = (xi - X_CONTRACT_1) / (X_CONTRACT_2 - X_CONTRACT_1)
            t_smooth = 10*t**3 - 15*t**4 + 6*t**5
            w.flat[i] = (H_IN/2)*(1-t_smooth) + (H_THROAT/2)*t_smooth
        elif xi <= X_EXPAND_1:
            w.flat[i] = H_THROAT / 2
        elif xi <= X_EXPAND_2:
            t = (xi - X_EXPAND_1) / (X_EXPAND_2 - X_EXPAND_1)
            t_smooth = 10*t**3 - 15*t**4 + 6*t**5
            w.flat[i] = (H_THROAT/2)*(1-t_smooth) + (H_IN/2)*t_smooth
        else:
            w.flat[i] = H_IN / 2
    
    w = w.reshape(x_np.shape)
    
    if isinstance(x, torch.Tensor):
        return torch.tensor(w, device=x.device, dtype=torch.float32)
    return w

def is_inside_channel(x, y):
    """Проверяет, находится ли точка внутри канала"""
    return torch.abs(y) <= channel_width(x)

# ============================================
# 2. Архитектура DeepONet
# ============================================
class DeepONet2D(nn.Module):
    def __init__(self, n_sensors=20, hidden_dim=64):
        super().__init__()
        # Branch: обрабатывает входную функцию (профиль скорости на входе)
        self.branch = nn.Sequential(
            nn.Linear(n_sensors, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
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
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Инициализация
        for net in [self.branch, self.trunk]:
            for layer in net:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
        
    def forward(self, sensor_vals, coords):
        b = self.branch(sensor_vals)    # (batch, hidden_dim)
        t = self.trunk(coords)          # (batch, N, hidden_dim)
        return torch.sum(b.unsqueeze(1) * t, dim=-1)  # (batch, N)


class NavierStokesDeepONet(nn.Module):
    def __init__(self, n_sensors=20):
        super().__init__()
        self.u_net = DeepONet2D(n_sensors, hidden_dim=64)
        self.v_net = DeepONet2D(n_sensors, hidden_dim=64)
        self.p_net = DeepONet2D(n_sensors, hidden_dim=64)
        
    def forward(self, sensor_vals, coords):
        u = self.u_net(sensor_vals, coords)
        v = self.v_net(sensor_vals, coords)
        p = self.p_net(sensor_vals, coords)
        return u, v, p

# ============================================
# 3. Physics-Informed Loss
# ============================================
def navier_stokes_loss(model, sensor_vals, coords):
    """Уравнения Навье-Стокса для несжимаемой жидкости"""
    coords = coords.detach().requires_grad_(True)
    
    u, v, p = model(sensor_vals, coords)
    
    # Градиенты
    ones = torch.ones_like(u)
    
    grad_u = torch.autograd.grad(u, coords, grad_outputs=ones, create_graph=True)[0]
    du_dx, du_dy = grad_u[..., 0], grad_u[..., 1]
    
    grad_v = torch.autograd.grad(v, coords, grad_outputs=ones, create_graph=True)[0]
    dv_dx, dv_dy = grad_v[..., 0], grad_v[..., 1]
    
    grad_p = torch.autograd.grad(p, coords, grad_outputs=ones, create_graph=True)[0]
    dp_dx, dp_dy = grad_p[..., 0], grad_p[..., 1]
    
    # Вторые производные
    d2u_dx2 = torch.autograd.grad(du_dx, coords, grad_outputs=ones, create_graph=True)[0][..., 0]
    d2u_dy2 = torch.autograd.grad(du_dy, coords, grad_outputs=ones, create_graph=True)[0][..., 1]
    d2v_dx2 = torch.autograd.grad(dv_dx, coords, grad_outputs=ones, create_graph=True)[0][..., 0]
    d2v_dy2 = torch.autograd.grad(dv_dy, coords, grad_outputs=ones, create_graph=True)[0][..., 1]
    
    # Уравнения Навье-Стокса
    momentum_x = u*du_dx + v*du_dy + dp_dx/RHO - NU*(d2u_dx2 + d2u_dy2)
    momentum_y = u*dv_dx + v*dv_dy + dp_dy/RHO - NU*(d2v_dx2 + d2v_dy2)
    continuity = du_dx + dv_dy
    
    return (torch.mean(momentum_x**2) + torch.mean(momentum_y**2) + 
            2.0 * torch.mean(continuity**2))

# ============================================
# 4. Генерация данных
# ============================================
def generate_constricted_data(n_samples=100, n_sensors=20):
    """
    Генерирует данные для течения в сужающейся трещине.
    Каждый сэмпл - разная скорость на входе.
    """
    print(f"📊 Генерация {n_samples} сэмплов...")
    
    y_sensors = np.linspace(-H_IN/2, H_IN/2, n_sensors, dtype=np.float32)
    
    all_sensors = np.zeros((n_samples, n_sensors), dtype=np.float32)
    all_coords = []
    all_u = []
    all_v = []
    all_p = []
    
    # Сетка для CFD-подобного решения (упрощенное)
    nx, ny = 40, 30
    x_grid = np.linspace(0, L, nx)
    
    for sample_idx in range(n_samples):
        # Случайная скорость на входе
        u_inlet = np.random.uniform(0.3, 2.0)
        
        # Сенсоры на входе (параболический профиль)
        sensor_vals = u_inlet * (1 - (2*y_sensors/H_IN)**2)
        sensor_vals += np.random.normal(0, 0.005, n_sensors)
        all_sensors[sample_idx] = sensor_vals.astype(np.float32)
        
        # Генерируем приближенное решение
        coords_list = []
        u_list = []
        v_list = []
        p_list = []
        
        for i, x in enumerate(x_grid):
            h = channel_width(x)
            y_grid_local = np.linspace(-h, h, ny)
            
            for y in y_grid_local:
                coords_list.append([x, y])
                
                # Приближенное решение:
                # u(y) ~ параболический профиль, масштабированный по ширине
                h_in = H_IN / 2
                h_local = channel_width(x)
                
                # Сохранение массы: u_max * h = const
                u_max_local = u_inlet * h_in / h_local
                
                # Параболический профиль
                u_val = u_max_local * (1 - (y/h_local)**2)
                
                # v-компонента (из уравнения неразрывности)
                if i > 0 and i < nx-1:
                    h_prev = channel_width(x_grid[i-1])
                    h_next = channel_width(x_grid[i+1])
                    dh_dx = (h_next - h_prev) / (x_grid[i+1] - x_grid[i-1])
                    v_val = u_val * y / h_local * dh_dx
                else:
                    v_val = 0.0
                
                # Давление (падает к выходу, ниже в горловине)
                p_val = u_inlet * (1 - x/L) * (h_local/h_in)
                
                u_list.append(u_val)
                v_list.append(v_val)
                p_list.append(p_val)
        
        all_coords.append(np.array(coords_list, dtype=np.float32))
        all_u.append(np.array(u_list, dtype=np.float32))
        all_v.append(np.array(v_list, dtype=np.float32))
        all_p.append(np.array(p_list, dtype=np.float32))
        
        if (sample_idx + 1) % 25 == 0:
            print(f"  Сгенерировано {sample_idx+1}/{n_samples} сэмплов")
    
    return (torch.FloatTensor(all_sensors),
            torch.FloatTensor(np.array(all_coords)),
            torch.FloatTensor(np.array(all_u)),
            torch.FloatTensor(np.array(all_v)),
            torch.FloatTensor(np.array(all_p)))

# ============================================
# 5. Подготовка данных
# ============================================
print("📊 Генерация данных для сужающейся трещины...")
sensors, coords, u_true, v_true, p_true = generate_constricted_data(100, 20)

# Разделение на train/test
train_size = 80
s_train, s_test = sensors[:train_size], sensors[train_size:]
c_train, c_test = coords[:train_size], coords[train_size:]
u_train, u_test = u_true[:train_size], u_true[train_size:]
v_train, v_test = v_true[:train_size], v_true[train_size:]
p_train, p_test = p_true[:train_size], p_true[train_size:]

print(f"✅ Train: {train_size}, Test: {len(s_test)}")

# ============================================
# 6. Обучение
# ============================================
print("🚀 Обучение DeepONet + PINN...")
model = NavierStokesDeepONet(n_sensors=20)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.7)

losses = {'total': [], 'data': [], 'pde': []}
best_loss = float('inf')

for epoch in range(1000):
    model.train()
    
    # Data loss
    u_pred, v_pred, p_pred = model(s_train, c_train)
    loss_data = (torch.mean((u_pred - u_train)**2) + 
                 torch.mean((v_pred - v_train)**2) + 
                 torch.mean((p_pred - p_train)**2))
    
    # PDE loss на случайных точках
    n_pde = 200
    # Генерируем точки внутри канала
    x_pde = torch.rand(s_train.shape[0], n_pde) * L
    y_pde = torch.rand(s_train.shape[0], n_pde) * H_IN - H_IN/2
    coords_pde = torch.stack([x_pde, y_pde], dim=-1)
    
    # Маскируем точки вне канала (быстро)
    h_pde = channel_width(x_pde)
    mask = torch.abs(y_pde) <= h_pde
    
    loss_pde = navier_stokes_loss(model, s_train, coords_pde)
    
    # Общая потеря
    loss = loss_data + 0.1 * loss_pde
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    
    losses['total'].append(loss.item())
    losses['data'].append(loss_data.item())
    losses['pde'].append(loss_pde.item())
    
    if loss.item() < best_loss:
        best_loss = loss.item()
        torch.save(model.state_dict(), f'{OUT_DIR}/best_model.pth')
    
    if epoch % 200 == 0:
        print(f"Epoch {epoch:4d} | Total: {loss.item():.4f} | Data: {loss_data.item():.4f} | PDE: {loss_pde.item():.4f}")

print("✅ Обучение завершено!")
model.load_state_dict(torch.load(f'{OUT_DIR}/best_model.pth'))

# ============================================
# 7. Визуализация
# ============================================
print("\n📈 Создание визуализации...")

# Тестовая сетка высокого разрешения
GRID_NX, GRID_NY = 150, 60
x_grid = np.linspace(0, L, GRID_NX)
y_grid = np.linspace(-H_IN/2, H_IN/2, GRID_NY)
X, Y = np.meshgrid(x_grid, y_grid)
grid_coords = torch.FloatTensor(np.stack([X.flatten(), Y.flatten()], axis=-1)).unsqueeze(0)

# Тестируем на новых входных скоростях
test_cases = [
    ('Низкая скорость', 0.5),
    ('Средняя скорость', 1.0),
    ('Высокая скорость', 1.8),
]

y_sensors = np.linspace(-H_IN/2, H_IN/2, 20)

fig, axes = plt.subplots(len(test_cases), 4, figsize=(20, 4*len(test_cases)))

for i, (name, u_in) in enumerate(test_cases):
    sensor_vals = u_in * (1 - (2*y_sensors/H_IN)**2)
    sensors_tensor = torch.FloatTensor(sensor_vals).unsqueeze(0)
    
    with torch.no_grad():
        u_pred, v_pred, p_pred = model(sensors_tensor, grid_coords)
        u_pred = u_pred.squeeze().numpy().reshape(GRID_NY, GRID_NX)
        v_pred = v_pred.squeeze().numpy().reshape(GRID_NY, GRID_NX)
        p_pred = p_pred.squeeze().numpy().reshape(GRID_NY, GRID_NX)
    
    # Маска геометрии
    mask = np.zeros_like(X, dtype=bool)
    for j in range(GRID_NX):
        mask[:, j] = np.abs(y_grid) <= channel_width(x_grid[j])
    
    u_pred[~mask] = np.nan
    v_pred[~mask] = np.nan
    p_pred[~mask] = np.nan
    speed = np.sqrt(u_pred**2 + v_pred**2)
    
    # Стены
    x_wall = np.linspace(0, L, 500)
    y_top = channel_width(x_wall)
    y_bottom = -channel_width(x_wall)
    
    # u-скорость
    im1 = axes[i, 0].pcolormesh(X, Y, u_pred, cmap='RdBu_r', shading='auto')
    axes[i, 0].plot(x_wall, y_top, 'k-', linewidth=1.5)
    axes[i, 0].plot(x_wall, y_bottom, 'k-', linewidth=1.5)
    axes[i, 0].set_title(f'{name}: u-velocity (inlet={u_in})')
    axes[i, 0].set_aspect('equal')
    plt.colorbar(im1, ax=axes[i, 0])
    
    # Магнитуда скорости
    im2 = axes[i, 1].pcolormesh(X, Y, speed, cmap='plasma', shading='auto')
    axes[i, 1].plot(x_wall, y_top, 'k-', linewidth=1.5)
    axes[i, 1].plot(x_wall, y_bottom, 'k-', linewidth=1.5)
    axes[i, 1].set_title(f'{name}: Speed')
    axes[i, 1].set_aspect('equal')
    plt.colorbar(im2, ax=axes[i, 1])
    
    # Давление
    im3 = axes[i, 2].pcolormesh(X, Y, p_pred, cmap='viridis', shading='auto')
    axes[i, 2].plot(x_wall, y_top, 'k-', linewidth=1.5)
    axes[i, 2].plot(x_wall, y_bottom, 'k-', linewidth=1.5)
    axes[i, 2].set_title(f'{name}: Pressure')
    axes[i, 2].set_aspect('equal')
    plt.colorbar(im3, ax=axes[i, 2])
    
    # Профиль скорости в горловине
    throat_idx = np.argmin(np.abs(x_grid - 1.0))
    inlet_idx = np.argmin(np.abs(x_grid - 0.3))
    
    y_valid = y_grid[~np.isnan(u_pred[:, throat_idx])]
    u_throat = u_pred[~np.isnan(u_pred[:, throat_idx]), throat_idx]
    u_inlet_profile = u_pred[~np.isnan(u_pred[:, inlet_idx]), inlet_idx]
    y_inlet_valid = y_grid[~np.isnan(u_pred[:, inlet_idx])]
    
    axes[i, 3].plot(u_inlet_profile, y_inlet_valid/H_IN*2, 'b-', label='Inlet', linewidth=2)
    axes[i, 3].plot(u_throat, y_valid/H_THROAT*2, 'r-', label='Throat', linewidth=2)
    axes[i, 3].set_xlabel('u [m/s]')
    axes[i, 3].set_ylabel('y/h (normalized)')
    axes[i, 3].set_title(f'{name}: Velocity Profiles')
    axes[i, 3].legend()
    axes[i, 3].grid(True, alpha=0.3)

plt.suptitle('DeepONet + PINN: Течение в сужающейся трещине (Навье-Стокс)', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/constricted_channel_results.png', dpi=200, bbox_inches='tight')
plt.close()

# График потерь
fig_loss, ax_loss = plt.subplots(figsize=(10, 5))
ax_loss.semilogy(losses['total'], 'k-', label='Total', linewidth=2)
ax_loss.semilogy(losses['data'], label='Data', alpha=0.7)
ax_loss.semilogy(losses['pde'], label='PDE', alpha=0.7)
ax_loss.set_xlabel('Epoch')
ax_loss.set_ylabel('Loss')
ax_loss.set_title('DeepONet + PINN Training History')
ax_loss.legend()
ax_loss.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/training_loss.png', dpi=150, bbox_inches='tight')
plt.close()

# Статистика
with torch.no_grad():
    u_test_pred, v_test_pred, p_test_pred = model(s_test, c_test)
    rel_error_u = torch.norm(u_test_pred - u_test) / torch.norm(u_test)
    rel_error_v = torch.norm(v_test_pred - v_test) / torch.norm(v_test)
    rel_error_p = torch.norm(p_test_pred - p_test) / torch.norm(p_test)

print("\n" + "="*60)
print("📊 Результаты для сужающейся трещины:")
print("="*60)
print(f"Относительная ошибка u: {rel_error_u:.4f}")
print(f"Относительная ошибка v: {rel_error_v:.4f}")
print(f"Относительная ошибка p: {rel_error_p:.4f}")
print(f"\n🎯 Геометрия:")
print(f"• Длина: {L} м")
print(f"• Вход: {H_IN} м → Горловина: {H_THROAT} м")
print(f"• Сужение: {X_CONTRACT_1}-{X_CONTRACT_2} м")
print(f"• Расширение: {X_EXPAND_1}-{X_EXPAND_2} м")
print(f"• Степень сужения: {H_IN/H_THROAT:.2f}x")
print(f"\n💡 DeepONet обучен для разных входных скоростей!")
print(f"📁 Результаты сохранены в: {OUT_DIR}/")
print("="*60)