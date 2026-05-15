import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
import gc
from matplotlib.colors import Normalize

# ============================================
# DeepONet + PINN с LBM-данными для сужающейся трещины
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

# Параметры LBM
LBM_NX = 200  # количество ячеек по x
LBM_NY = 80   # количество ячеек по y
LBM_TAU = 1.0 / 3.0 + NU * 3.0 * (LBM_NY / H_IN)  # время релаксации
LBM_STEPS = 5000  # шагов симуляции

# Директория для сохранения
OUT_DIR = "deeponet_lbm_constricted"
os.makedirs(OUT_DIR, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используется устройство: {device}")

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()

# ============================================
# 1. Геометрия сужающейся трещины
# ============================================
def channel_width(x):
    """Полуширина канала в точке x"""
    if isinstance(x, torch.Tensor):
        was_tensor = True
        x_np = x.detach().cpu().numpy()
        device_orig = x.device
    else:
        was_tensor = False
        x_np = np.asarray(x, dtype=np.float64)
    
    w = np.full_like(x_np, H_IN/2, dtype=np.float64)
    
    # Сужение
    mask_contract = (x_np > X_CONTRACT_1) & (x_np <= X_CONTRACT_2)
    if mask_contract.any():
        t_contract = (x_np[mask_contract] - X_CONTRACT_1) / (X_CONTRACT_2 - X_CONTRACT_1)
        t_smooth_contract = np.sin(np.pi * t_contract / 2)**2
        w[mask_contract] = (H_IN/2)*(1-t_smooth_contract) + (H_THROAT/2)*t_smooth_contract
    
    # Горло
    mask_throat = (x_np > X_CONTRACT_2) & (x_np <= X_EXPAND_1)
    if mask_throat.any():
        w[mask_throat] = H_THROAT / 2
    
    # Расширение
    mask_expand = (x_np > X_EXPAND_1) & (x_np <= X_EXPAND_2)
    if mask_expand.any():
        t_expand = (x_np[mask_expand] - X_EXPAND_1) / (X_EXPAND_2 - X_EXPAND_1)
        t_smooth_expand = np.sin(np.pi * t_expand / 2)**2
        w[mask_expand] = (H_THROAT/2)*(1-t_smooth_expand) + (H_IN/2)*t_smooth_expand
    
    if was_tensor:
        return torch.tensor(w, device=device_orig, dtype=torch.float32)
    return w


# ============================================
# 2. LBM Симулятор (D2Q9)
# ============================================
class LBMSimulator:
    """Lattice Boltzmann Method симулятор D2Q9"""
    
    def __init__(self, nx, ny, tau, u_inlet_max):
        self.nx = nx
        self.ny = ny
        self.tau = tau
        self.omega = 1.0 / tau
        self.u_inlet_max = u_inlet_max
        
        # Веса D2Q9
        self.w = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
        
        # Векторы скоростей D2Q9
        self.c = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
                           [1, 1], [-1, 1], [-1, -1], [1, -1]])
        
        # Противоположные направления
        self.opposite = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
        
        # Инициализация
        self.f = np.zeros((9, ny, nx))
        self.f_eq = np.zeros((9, ny, nx))
        self.rho = np.ones((ny, nx))
        self.ux = np.zeros((ny, nx))
        self.uy = np.zeros((ny, nx))
        
        # Создаем маску геометрии
        self.create_geometry()
        
        # Инициализация равновесным распределением
        self.init_equilibrium()
    
    def create_geometry(self):
        """Создание маски геометрии"""
        self.solid = np.zeros((self.ny, self.nx), dtype=bool)
        
        dx = L / self.nx
        for i in range(self.nx):
            x = i * dx
            h = channel_width(x)
            for j in range(self.ny):
                y = (j - self.ny/2) * (H_IN / self.ny)
                if abs(y) > h:
                    self.solid[j, i] = True
    
    def init_equilibrium(self):
        """Инициализация равновесным распределением"""
        for k in range(9):
            cu = self.c[k, 0] * self.ux + self.c[k, 1] * self.uy
            self.f[k] = self.w[k] * self.rho * (1 + 3*cu + 4.5*cu**2 - 1.5*(self.ux**2 + self.uy**2))
    
    def equilibrium(self, rho, ux, uy):
        """Вычисление равновесного распределения"""
        feq = np.zeros((9, self.ny, self.nx))
        u2 = ux**2 + uy**2
        for k in range(9):
            cu = self.c[k, 0] * ux + self.c[k, 1] * uy
            feq[k] = self.w[k] * rho * (1 + 3*cu + 4.5*cu**2 - 1.5*u2)
        return feq
    
    def compute_macroscopic(self):
        """Вычисление макроскопических величин"""
        self.rho = np.sum(self.f, axis=0)
        self.ux = np.sum(self.f * self.c[:, 0].reshape(-1, 1, 1), axis=0) / (self.rho + 1e-8)
        self.uy = np.sum(self.f * self.c[:, 1].reshape(-1, 1, 1), axis=0) / (self.rho + 1e-8)
    
    def apply_bounceback(self):
        """Граничные условия bounce-back на стенках"""
        for k in range(9):
            opp = self.opposite[k]
            # Сдвигаем на -c_k
            f_shifted = np.roll(np.roll(self.f[k], -self.c[k, 1], axis=0), -self.c[k, 0], axis=1)
            # Применяем bounce-back где твердые стенки
            self.f[opp][self.solid] = f_shifted[self.solid]
    
    def inlet_boundary(self):
        """Граничные условия на входе (параболический профиль)"""
        inlet_x = 0
        for j in range(self.ny):
            if not self.solid[j, inlet_x]:
                y = (j - self.ny/2) * (H_IN / self.ny)
                h_local = channel_width(0)
                u_profile = self.u_inlet_max * (1 - (y/h_local)**2)
                
                # Zou-He boundary condition
                rho_in = (self.f[0, j, inlet_x] + self.f[2, j, inlet_x] + self.f[4, j, inlet_x] + 
                         2*(self.f[3, j, inlet_x] + self.f[6, j, inlet_x] + self.f[7, j, inlet_x])) / (1 - u_profile)
                
                self.f[1, j, inlet_x] = self.f[3, j, inlet_x] + 2/3 * rho_in * u_profile
                self.f[5, j, inlet_x] = self.f[7, j, inlet_x] + 1/6 * rho_in * u_profile
                self.f[8, j, inlet_x] = self.f[6, j, inlet_x] + 1/6 * rho_in * u_profile
    
    def outlet_boundary(self):
        """Граничные условия на выходе (постоянное давление)"""
        outlet_x = -1
        p_out = 1.0
        for j in range(self.ny):
            if not self.solid[j, outlet_x]:
                ux_out = -1 + (self.f[0, j, outlet_x] + self.f[1, j, outlet_x] + self.f[3, j, outlet_x] + 
                               2*(self.f[2, j, outlet_x] + self.f[5, j, outlet_x] + self.f[6, j, outlet_x])) / p_out
                
                self.f[4, j, outlet_x] = self.f[2, j, outlet_x] - 2/3 * p_out * ux_out
                self.f[7, j, outlet_x] = self.f[5, j, outlet_x] - 1/6 * p_out * ux_out
                self.f[8, j, outlet_x] = self.f[6, j, outlet_x] - 1/6 * p_out * ux_out
    
    def collide_and_stream(self):
        """Шаг столкновения и распространения"""
        # Вычисляем равновесное распределение
        feq = self.equilibrium(self.rho, self.ux, self.uy)
        
        # Столкновение (BGK)
        for k in range(9):
            self.f[k] = self.f[k] - self.omega * (self.f[k] - feq[k])
        
        # Распространение
        for k in range(9):
            self.f[k] = np.roll(np.roll(self.f[k], self.c[k, 1], axis=0), self.c[k, 0], axis=1)
    
    def step(self):
        """Один шаг LBM"""
        self.compute_macroscopic()
        self.apply_bounceback()
        self.inlet_boundary()
        self.outlet_boundary()
        self.collide_and_stream()
    
    def run(self, n_steps, print_interval=500):
        """Запуск симуляции"""
        print(f"Запуск LBM симуляции (u_inlet={self.u_inlet_max:.2f}, {n_steps} шагов)...")
        
        for step in range(n_steps):
            self.step()
            
            if (step + 1) % print_interval == 0:
                max_u = np.max(np.abs(self.ux))
                print(f"  Шаг {step+1}/{n_steps}, max|u| = {max_u:.4f}")
        
        # Финальное вычисление
        self.compute_macroscopic()
        
        # Вычисление давления (p = rho * c_s^2)
        cs2 = 1.0 / 3.0
        p = (self.rho - 1.0) * cs2
        
        return self.ux.copy(), self.uy.copy(), p


# ============================================
# 3. Архитектура DeepONet
# ============================================
class ImprovedDeepONet2D(nn.Module):
    def __init__(self, n_sensors=20, hidden_dim=128, n_layers=3):
        super().__init__()
        
        self.fourier_features = 32
        self.trunk_input_dim = 2 + 4 * self.fourier_features
        
        # Branch network
        branch_layers = []
        input_dim = n_sensors
        
        for i in range(n_layers):
            branch_layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
            branch_layers.append(nn.LayerNorm(hidden_dim))
            branch_layers.append(nn.GELU())
            if i < n_layers - 1:
                branch_layers.append(nn.Dropout(0.1))
        
        self.branch = nn.Sequential(*branch_layers)
        
        # Trunk network
        trunk_layers = []
        trunk_layers.append(nn.Linear(self.trunk_input_dim, hidden_dim))
        trunk_layers.append(nn.LayerNorm(hidden_dim))
        trunk_layers.append(nn.GELU())
        
        for i in range(n_layers - 1):
            trunk_layers.append(nn.Linear(hidden_dim, hidden_dim))
            trunk_layers.append(nn.LayerNorm(hidden_dim))
            trunk_layers.append(nn.GELU())
            if i < n_layers - 2:
                trunk_layers.append(nn.Dropout(0.05))
        
        self.trunk = nn.Sequential(*trunk_layers)
        
        # Final layers
        self.final = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.01)
    
    def _add_fourier_features(self, coords):
        batch_size, n_points, _ = coords.shape
        
        freqs = torch.exp(torch.linspace(0, np.log(8), self.fourier_features, device=coords.device))
        
        x = coords[..., 0:1]
        y = coords[..., 1:2]
        
        x_ff = x * freqs.view(1, 1, -1)
        y_ff = y * freqs.view(1, 1, -1)
        
        ff = torch.cat([torch.sin(x_ff), torch.cos(x_ff), torch.sin(y_ff), torch.cos(y_ff)], dim=-1)
        
        return torch.cat([coords, ff], dim=-1)
    
    def forward(self, sensor_vals, coords):
        batch_size, n_points, _ = coords.shape
        
        coords_ff = self._add_fourier_features(coords)
        
        b = self.branch(sensor_vals)
        t = self.trunk(coords_ff)
        
        b_expanded = b.unsqueeze(1)
        product = b_expanded * t
        
        out = self.final(product).squeeze(-1)
        
        return out

class NavierStokesDeepONet(nn.Module):
    def __init__(self, n_sensors=20):
        super().__init__()
        self.u_net = ImprovedDeepONet2D(n_sensors, hidden_dim=128, n_layers=3)
        self.v_net = ImprovedDeepONet2D(n_sensors, hidden_dim=128, n_layers=3)
        self.p_net = ImprovedDeepONet2D(n_sensors, hidden_dim=128, n_layers=3)
    
    def forward(self, sensor_vals, coords):
        u = self.u_net(sensor_vals, coords)
        v = self.v_net(sensor_vals, coords)
        p = self.p_net(sensor_vals, coords)
        return u, v, p

# ============================================
# 4. Physics-Informed Loss
# ============================================
def navier_stokes_loss(model, sensor_vals, coords):
    coords = coords.detach().requires_grad_(True)
    u, v, p = model(sensor_vals, coords)
    
    ones = torch.ones_like(u)
    
    grad_u = torch.autograd.grad(u, coords, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    du_dx, du_dy = grad_u[..., 0], grad_u[..., 1]
    
    grad_v = torch.autograd.grad(v, coords, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    dv_dx, dv_dy = grad_v[..., 0], grad_v[..., 1]
    
    grad_p = torch.autograd.grad(p, coords, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    dp_dx, dp_dy = grad_p[..., 0], grad_p[..., 1]
    
    du_dx_grad = torch.autograd.grad(du_dx, coords, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    d2u_dx2 = du_dx_grad[..., 0]
    d2u_dy2 = torch.autograd.grad(du_dy, coords, grad_outputs=ones, create_graph=True, retain_graph=True)[0][..., 1]
    
    dv_dx_grad = torch.autograd.grad(dv_dx, coords, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    d2v_dx2 = dv_dx_grad[..., 0]
    d2v_dy2 = torch.autograd.grad(dv_dy, coords, grad_outputs=ones, create_graph=True, retain_graph=True)[0][..., 1]
    
    momentum_x = u*du_dx + v*du_dy + dp_dx/RHO - NU*(d2u_dx2 + d2u_dy2)
    momentum_y = u*dv_dx + v*dv_dy + dp_dy/RHO - NU*(d2v_dx2 + d2v_dy2)
    continuity = du_dx + dv_dy
    
    loss = torch.mean(momentum_x**2) + torch.mean(momentum_y**2) + 2.0 * torch.mean(continuity**2)
    
    del grad_u, grad_v, grad_p, du_dx_grad
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return loss

# ============================================
# 5. Генерация данных через LBM
# ============================================
def generate_lbm_data(n_samples=100, n_sensors=30):
    """Генерация данных с помощью LBM симуляций"""
    print(f"Генерация {n_samples} сэмплов через LBM...")
    
    y_sensors = np.linspace(-H_IN/2, H_IN/2, n_sensors, dtype=np.float32)
    all_sensors = np.zeros((n_samples, n_sensors), dtype=np.float32)
    all_coords = []
    all_u = []
    all_v = []
    all_p = []
    
    # Сетка для выходных данных
    nx_out, ny_out = 50, 30
    x_grid = np.linspace(0, L, nx_out)
    
    for sample_idx in range(n_samples):
        u_inlet = np.random.uniform(0.3, 2.0)
        
        print(f"  Сэмпл {sample_idx+1}/{n_samples} (u_inlet={u_inlet:.2f})...")
        
        # Запуск LBM
        lbm = LBMSimulator(LBM_NX, LBM_NY, LBM_TAU, u_inlet)
        
        # Адаптивное количество шагов
        if sample_idx == 0:
            n_steps = LBM_STEPS
        else:
            n_steps = LBM_STEPS // 2  # Меньше шагов для скорости
        
        u_lbm, v_lbm, p_lbm = lbm.run(n_steps, print_interval=1000)
        
        # Интерполяция LBM результатов на выходную сетку
        coords_list = []
        u_list = []
        v_list = []
        p_list = []
        
        for i, x in enumerate(x_grid):
            h = max(channel_width(x), 0.01)
            y_grid_local = np.linspace(-h*0.95, h*0.95, ny_out)
            
            for y in y_grid_local:
                coords_list.append([x, y])
                
                # Интерполяция из LBM сетки
                lbm_x_idx = int(x / L * LBM_NX)
                lbm_y_idx = int((y + H_IN/2) / H_IN * LBM_NY)
                
                # Билинейная интерполяция
                x0 = min(lbm_x_idx, LBM_NX-2)
                y0 = min(lbm_y_idx, LBM_NY-2)
                x1 = x0 + 1
                y1 = y0 + 1
                
                fx = x * LBM_NX / L - x0
                fy = (y + H_IN/2) * LBM_NY / H_IN - y0
                
                u_val = (u_lbm[y0, x0] * (1-fx) * (1-fy) + u_lbm[y0, x1] * fx * (1-fy) +
                         u_lbm[y1, x0] * (1-fx) * fy + u_lbm[y1, x1] * fx * fy)
                v_val = (v_lbm[y0, x0] * (1-fx) * (1-fy) + v_lbm[y0, x1] * fx * (1-fy) +
                         v_lbm[y1, x0] * (1-fx) * fy + v_lbm[y1, x1] * fx * fy)
                p_val = (p_lbm[y0, x0] * (1-fx) * (1-fy) + p_lbm[y0, x1] * fx * (1-fy) +
                         p_lbm[y1, x0] * (1-fx) * fy + p_lbm[y1, x1] * fx * fy)
                
                u_list.append(u_val)
                v_list.append(v_val)
                p_list.append(p_val)
        
        # Сенсорные данные (профиль скорости на входе из LBM)
        for j, y_s in enumerate(y_sensors):
            lbm_y_idx = int((y_s + H_IN/2) / H_IN * LBM_NY)
            lbm_y_idx = min(max(lbm_y_idx, 0), LBM_NY-1)
            sensor_val = u_lbm[lbm_y_idx, 0]  # Берем скорость на входе
            all_sensors[sample_idx, j] = sensor_val
        
        all_coords.append(np.array(coords_list, dtype=np.float32))
        all_u.append(np.array(u_list, dtype=np.float32))
        all_v.append(np.array(v_list, dtype=np.float32))
        all_p.append(np.array(p_list, dtype=np.float32))
        
        # Очистка памяти
        del lbm, u_lbm, v_lbm, p_lbm
        gc.collect()
    
    # Padding
    max_len = max(len(c) for c in all_coords)
    for i in range(len(all_coords)):
        if len(all_coords[i]) < max_len:
            pad_len = max_len - len(all_coords[i])
            all_coords[i] = np.pad(all_coords[i], ((0, pad_len), (0, 0)), mode='edge')
            all_u[i] = np.pad(all_u[i], (0, pad_len), mode='edge')
            all_v[i] = np.pad(all_v[i], (0, pad_len), mode='edge')
            all_p[i] = np.pad(all_p[i], (0, pad_len), mode='edge')
    
    return (torch.FloatTensor(all_sensors),
            torch.FloatTensor(np.array(all_coords)),
            torch.FloatTensor(np.array(all_u)),
            torch.FloatTensor(np.array(all_v)),
            torch.FloatTensor(np.array(all_p)))

# ============================================
# 6. Генерация точек для PDE loss
# ============================================
def generate_interior_points(batch_size, n_points, device):
    x = torch.rand(batch_size, n_points, device=device) * L
    y = (torch.rand(batch_size, n_points, device=device) - 0.5) * H_IN
    
    h = channel_width(x)
    mask = torch.abs(y) > h
    y[mask] = torch.sign(y[mask]) * h[mask] * 0.99
    
    return torch.stack([x, y], dim=-1)

# ============================================
# 7. Функции визуализации
# ============================================
def plot_training_history(losses, save_path):
    """График истории обучения"""
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    axes[0].semilogy(losses['total'], 'k-', label='Total Loss', linewidth=2)
    axes[0].semilogy(losses['data'], 'b-', label='Data Loss', alpha=0.7)
    axes[0].semilogy(losses['pde'], 'r-', label='PDE Loss', alpha=0.7)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss (log scale)', fontsize=12)
    axes[0].set_title('Training History', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    if 'data_u' in losses:
        axes[1].semilogy(losses['data_u'], label='Data u', alpha=0.7)
        axes[1].semilogy(losses['data_v'], label='Data v', alpha=0.7)
        axes[1].semilogy(losses['data_p'], label='Data p', alpha=0.7)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss (log scale)', fontsize=12)
    axes[1].set_title('Data Loss Components', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('DeepONet + PINN (LBM data): Training Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_results_comprehensive(X, Y, u_pred_2d, v_pred_2d, p_pred_2d, 
                              x_wall, y_top, y_bottom, y_grid, save_path, title=""):
    """Комплексная визуализация результатов"""
    
    speed = np.sqrt(u_pred_2d**2 + v_pred_2d**2)
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    
    # u-скорость
    im1 = axes[0, 0].pcolormesh(X, Y, u_pred_2d, cmap='RdBu_r', shading='auto')
    axes[0, 0].plot(x_wall, y_top, 'k-', linewidth=1.5)
    axes[0, 0].plot(x_wall, y_bottom, 'k-', linewidth=1.5)
    axes[0, 0].set_title('u-velocity', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    axes[0, 0].set_aspect('equal')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Speed
    im2 = axes[0, 1].pcolormesh(X, Y, speed, cmap='plasma', shading='auto')
    axes[0, 1].plot(x_wall, y_top, 'k-', linewidth=1.5)
    axes[0, 1].plot(x_wall, y_bottom, 'k-', linewidth=1.5)
    axes[0, 1].set_title('Speed', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    axes[0, 1].set_aspect('equal')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Давление
    im3 = axes[0, 2].pcolormesh(X, Y, p_pred_2d, cmap='viridis', shading='auto')
    axes[0, 2].plot(x_wall, y_top, 'k-', linewidth=1.5)
    axes[0, 2].plot(x_wall, y_bottom, 'k-', linewidth=1.5)
    axes[0, 2].set_title('Pressure', fontsize=12, fontweight='bold')
    axes[0, 2].set_xlabel('x')
    axes[0, 2].set_ylabel('y')
    axes[0, 2].set_aspect('equal')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # v-скорость
    im4 = axes[0, 3].pcolormesh(X, Y, v_pred_2d, cmap='RdBu_r', shading='auto')
    axes[0, 3].plot(x_wall, y_top, 'k-', linewidth=1.5)
    axes[0, 3].plot(x_wall, y_bottom, 'k-', linewidth=1.5)
    axes[0, 3].set_title('v-velocity', fontsize=12, fontweight='bold')
    axes[0, 3].set_xlabel('x')
    axes[0, 3].set_ylabel('y')
    axes[0, 3].set_aspect('equal')
    plt.colorbar(im4, ax=axes[0, 3])
    
    # Профили скорости
    throat_idx = np.argmin(np.abs(x_grid - 1.0))
    inlet_idx = np.argmin(np.abs(x_grid - 0.2))
    
    valid_inlet = ~np.isnan(u_pred_2d[:, inlet_idx])
    if valid_inlet.any():
        y_inlet = y_grid[valid_inlet]
        axes[1, 0].plot(u_pred_2d[valid_inlet, inlet_idx], y_inlet, 'b-', label='Predicted', linewidth=2.5)
        axes[1, 0].set_title('Velocity Profile at Inlet (x=0.2)', fontsize=11, fontweight='bold')
        axes[1, 0].set_xlabel('u-velocity', fontsize=10)
        axes[1, 0].set_ylabel('y', fontsize=10)
        axes[1, 0].legend(fontsize=9)
        axes[1, 0].grid(True, alpha=0.3)
    
    valid_throat = ~np.isnan(u_pred_2d[:, throat_idx])
    if valid_throat.any():
        y_throat = y_grid[valid_throat]
        axes[1, 1].plot(u_pred_2d[valid_throat, throat_idx], y_throat, 'r-', label='Predicted', linewidth=2.5)
        axes[1, 1].set_title('Velocity Profile at Throat (x=1.0)', fontsize=11, fontweight='bold')
        axes[1, 1].set_xlabel('u-velocity', fontsize=10)
        axes[1, 1].set_ylabel('y', fontsize=10)
        axes[1, 1].legend(fontsize=9)
        axes[1, 1].grid(True, alpha=0.3)
    
    # Поле скорости
    skip = 4
    mask_quiver = ~np.isnan(u_pred_2d[::skip, ::skip])
    X_sub = X[::skip, ::skip][mask_quiver]
    Y_sub = Y[::skip, ::skip][mask_quiver]
    u_sub = u_pred_2d[::skip, ::skip][mask_quiver]
    v_sub = v_pred_2d[::skip, ::skip][mask_quiver]
    
    axes[1, 2].plot(x_wall, y_top, 'k-', linewidth=1.5)
    axes[1, 2].plot(x_wall, y_bottom, 'k-', linewidth=1.5)
    if len(X_sub) > 0:
        axes[1, 2].quiver(X_sub, Y_sub, u_sub, v_sub, scale=30, width=0.003)
    axes[1, 2].set_title('Velocity Field', fontsize=12, fontweight='bold')
    axes[1, 2].set_xlabel('x', fontsize=11)
    axes[1, 2].set_ylabel('y', fontsize=11)
    axes[1, 2].set_aspect('equal')
    axes[1, 2].grid(True, alpha=0.3)
    
    # Давление по центру
    center_idx = np.argmin(np.abs(y_grid))
    valid_center = ~np.isnan(p_pred_2d[center_idx, :])
    if valid_center.any():
        x_center = x_grid[valid_center]
        p_center = p_pred_2d[center_idx, valid_center]
        axes[1, 3].plot(x_center, p_center, 'b-', linewidth=2.5)
        axes[1, 3].set_title('Pressure along Centerline', fontsize=12, fontweight='bold')
        axes[1, 3].set_xlabel('x', fontsize=11)
        axes[1, 3].set_ylabel('Pressure', fontsize=11)
        axes[1, 3].grid(True, alpha=0.3)
    
    plt.suptitle(f'DeepONet + PINN (LBM data): {title}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================
# ОСНОВНОЙ КОД
# ============================================

# Генерация данных через LBM
print("Генерация данных через LBM...")
n_samples = 50  # Меньше сэмплов из-за времени LBM
sensors, coords, u_true, v_true, p_true = generate_lbm_data(n_samples, 25)

# Разделение
train_size = int(0.8 * n_samples)
s_train, s_test = sensors[:train_size], sensors[train_size:]
c_train, c_test = coords[:train_size], coords[train_size:]
u_train, u_test = u_true[:train_size], u_true[train_size:]
v_train, v_test = v_true[:train_size], v_true[train_size:]
p_train, p_test = p_true[:train_size], p_true[train_size:]

print(f"\nTrain: {train_size}, Test: {len(s_test)}")
print(f"Форма данных: coords={c_train.shape}, u={u_train.shape}")

# Статистика
u_mean, u_std = u_train.mean(), u_train.std()
v_mean, v_std = v_train.mean(), v_train.std()
p_mean, p_std = p_train.mean(), p_train.std()
print(f"Статистика: u({u_mean:.3f}±{u_std:.3f}), v({v_mean:.3f}±{v_std:.3f}), p({p_mean:.3f}±{p_std:.3f})")

# Переносим на устройство
s_train = s_train.to(device)
c_train = c_train.to(device)
u_train = u_train.to(device)
v_train = v_train.to(device)
p_train = p_train.to(device)

# Обучение
print("\nОбучение DeepONet + PINN на LBM данных...")
model = NavierStokesDeepONet(n_sensors=25).to(device)

optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=200, T_mult=2)

losses = {'total': [], 'data': [], 'pde': [], 'data_u': [], 'data_v': [], 'data_p': []}
best_loss = float('inf')
best_epoch = 0

n_epochs = 2000
weights = {'u': 1.0, 'v': 5.0, 'p': 0.5}
mini_batch_size = 10  # Меньше из-за меньшего количества данных

for epoch in range(n_epochs):
    model.train()
    
    total_loss = 0
    total_loss_data = 0
    total_loss_pde = 0
    total_loss_data_u = 0
    total_loss_data_v = 0
    total_loss_data_p = 0
    
    indices = torch.randperm(train_size)
    
    for start_idx in range(0, train_size, mini_batch_size):
        end_idx = min(start_idx + mini_batch_size, train_size)
        batch_indices = indices[start_idx:end_idx]
        
        s_batch = s_train[batch_indices]
        c_batch = c_train[batch_indices]
        u_batch = u_train[batch_indices]
        v_batch = v_train[batch_indices]
        p_batch = p_train[batch_indices]
        
        u_pred, v_pred, p_pred = model(s_batch, c_batch)
        loss_u = weights['u'] * torch.mean((u_pred - u_batch)**2)
        loss_v = weights['v'] * torch.mean((v_pred - v_batch)**2)
        loss_p = weights['p'] * torch.mean((p_pred - p_batch)**2)
        loss_data = (loss_u + loss_v + loss_p) / (end_idx - start_idx) * train_size
        
        n_pde = 200
        coords_pde = generate_interior_points(len(batch_indices), n_pde, device)
        loss_pde = navier_stokes_loss(model, s_batch, coords_pde) / (end_idx - start_idx) * train_size
        
        if epoch < 500:
            pde_weight = 0.01
        elif epoch < 1000:
            pde_weight = 0.05
        else:
            pde_weight = 0.1
        
        loss = loss_data + pde_weight * loss_pde
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        total_loss += loss.item()
        total_loss_data += loss_data.item()
        total_loss_pde += loss_pde.item()
        total_loss_data_u += loss_u.item()
        total_loss_data_v += loss_v.item()
        total_loss_data_p += loss_p.item()
        
        del u_pred, v_pred, p_pred, loss, loss_data, loss_pde
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    num_batches = (train_size + mini_batch_size - 1) // mini_batch_size
    losses['total'].append(total_loss / num_batches)
    losses['data'].append(total_loss_data / num_batches)
    losses['pde'].append(total_loss_pde / num_batches)
    losses['data_u'].append(total_loss_data_u / num_batches)
    losses['data_v'].append(total_loss_data_v / num_batches)
    losses['data_p'].append(total_loss_data_p / num_batches)
    
    if epoch % 50 == 0:
        scheduler.step()
    
    current_loss = losses['total'][-1]
    if current_loss < best_loss:
        best_loss = current_loss
        best_epoch = epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_loss,
        }, f'{OUT_DIR}/best_model.pth')
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d}/{n_epochs} | Total: {losses['total'][-1]:.6f} | "
              f"Data: {losses['data'][-1]:.4f} (u:{losses['data_u'][-1]:.4f}, v:{losses['data_v'][-1]:.4f}, p:{losses['data_p'][-1]:.4f}) | "
              f"PDE: {losses['pde'][-1]:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

print(f"\nОбучение завершено! Лучшая модель на эпохе {best_epoch}")
checkpoint = torch.load(f'{OUT_DIR}/best_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

# ============================================
# ВИЗУАЛИЗАЦИЯ
# ============================================
print("\nСоздание визуализации...")

# График обучения
plot_training_history(losses, f'{OUT_DIR}/training_history.png')

# Сетка для визуализации
GRID_NX, GRID_NY = 150, 80
x_grid = np.linspace(0, L, GRID_NX)
y_grid = np.linspace(-H_IN/2, H_IN/2, GRID_NY)
X, Y = np.meshgrid(x_grid, y_grid)
grid_coords = torch.FloatTensor(np.stack([X.flatten(), Y.flatten()], axis=-1)).unsqueeze(0).to(device)

# Стены канала
x_wall = np.linspace(0, L, 500)
y_top = channel_width(x_wall)
y_bottom = -channel_width(x_wall)

# Визуализация для нескольких скоростей
test_speeds = [0.5, 1.0, 1.5]

for u_in in test_speeds:
    print(f"  Визуализация для u_inlet = {u_in}...")
    
    y_sensors = np.linspace(-H_IN/2, H_IN/2, 25)
    sensor_vals = u_in * (1 - (2*y_sensors/H_IN)**2)
    sensors_tensor = torch.FloatTensor(sensor_vals).unsqueeze(0).to(device)
    
    with torch.no_grad():
        chunk_size = GRID_NX * GRID_NY // 4
        u_chunks = []
        v_chunks = []
        p_chunks = []
        
        for j in range(0, GRID_NX * GRID_NY, chunk_size):
            end_j = min(j + chunk_size, GRID_NX * GRID_NY)
            coords_chunk = grid_coords[:, j:end_j, :]
            u_c, v_c, p_c = model(sensors_tensor, coords_chunk)
            u_chunks.append(u_c.cpu())
            v_chunks.append(v_c.cpu())
            p_chunks.append(p_c.cpu())
        
        u_pred = torch.cat(u_chunks, dim=1).squeeze().numpy().reshape(GRID_NY, GRID_NX)
        v_pred = torch.cat(v_chunks, dim=1).squeeze().numpy().reshape(GRID_NY, GRID_NX)
        p_pred = torch.cat(p_chunks, dim=1).squeeze().numpy().reshape(GRID_NY, GRID_NX)
    
    # Маска геометрии
    mask = np.zeros_like(X, dtype=bool)
    for j in range(GRID_NX):
        mask[:, j] = np.abs(y_grid) <= channel_width(x_grid[j])
    
    u_pred = np.where(mask, u_pred, np.nan)
    v_pred = np.where(mask, v_pred, np.nan)
    p_pred = np.where(mask, p_pred, np.nan)
    
    plot_results_comprehensive(X, Y, u_pred, v_pred, p_pred, 
                              x_wall, y_top, y_bottom, y_grid,
                              f'{OUT_DIR}/results_u{u_in:.1f}.png',
                              f'u_inlet = {u_in}')

# Оценка на тестовых данных
print("\nОценка на тестовых данных...")
with torch.no_grad():
    s_test = s_test.to(device)
    c_test = c_test.to(device)
    u_test = u_test.to(device)
    v_test = v_test.to(device)
    p_test = p_test.to(device)
    
    test_batch_size = 5
    u_preds = []
    v_preds = []
    p_preds = []
    
    for j in range(0, len(s_test), test_batch_size):
        end_j = min(j + test_batch_size, len(s_test))
        u_p, v_p, p_p = model(s_test[j:end_j], c_test[j:end_j])
        u_preds.append(u_p)
        v_preds.append(v_p)
        p_preds.append(p_p)
    
    u_test_pred = torch.cat(u_preds, dim=0)
    v_test_pred = torch.cat(v_preds, dim=0)
    p_test_pred = torch.cat(p_preds, dim=0)
    
    rel_error_u = torch.norm(u_test_pred - u_test) / (torch.norm(u_test) + 1e-8)
    rel_error_v = torch.norm(v_test_pred - v_test) / (torch.norm(v_test) + 1e-8)
    rel_error_p = torch.norm(p_test_pred - p_test) / (torch.norm(p_test) + 1e-8)
    
    r2_u = 1 - torch.sum((u_test - u_test_pred)**2) / (torch.sum((u_test - torch.mean(u_test))**2) + 1e-8)
    r2_v = 1 - torch.sum((v_test - v_test_pred)**2) / (torch.sum((v_test - torch.mean(v_test))**2) + 1e-8)
    r2_p = 1 - torch.sum((p_test - p_test_pred)**2) / (torch.sum((p_test - torch.mean(p_test))**2) + 1e-8)
    
    mse_u = torch.mean((u_test_pred - u_test)**2).item()
    mse_v = torch.mean((v_test_pred - v_test)**2).item()
    mse_p = torch.mean((p_test_pred - p_test)**2).item()

results_text = f"""
{'='*60}
РЕЗУЛЬТАТЫ ОБУЧЕНИЯ DeepONet + PINN (LBM data)
{'='*60}

Финальные значения Loss:
   * Total Loss: {losses['total'][-1]:.4f}
   * Data Loss: {losses['data'][-1]:.4f}
   * PDE Loss: {losses['pde'][-1]:.4f}

Относительные L2 ошибки:
   * u-velocity: {rel_error_u:.4f} ({rel_error_u*100:.2f}%)
   * v-velocity: {rel_error_v:.4f} ({rel_error_v*100:.2f}%)
   * pressure:   {rel_error_p:.4f} ({rel_error_p*100:.2f}%)

Среднеквадратичные ошибки (MSE):
   * u-velocity: {mse_u:.6f}
   * v-velocity: {mse_v:.6f}
   * pressure:   {mse_p:.6f}

R² коэффициенты детерминации:
   * u-velocity: {r2_u:.4f}
   * v-velocity: {r2_v:.4f}
   * pressure:   {r2_p:.4f}

Параметры LBM:
   * Сетка: {LBM_NX}x{LBM_NY}
   * tau = {LBM_TAU:.4f}
   * Шагов симуляции: {LBM_STEPS}

Сохраненные файлы:
   * Модель: {OUT_DIR}/best_model.pth
   * Графики: {OUT_DIR}/results_*.png
   * История обучения: {OUT_DIR}/training_history.png
"""

print(results_text)

with open(f'{OUT_DIR}/results_summary.txt', 'w', encoding='utf-8') as f:
    f.write(results_text)

print(f"\nВсе результаты сохранены в папку: {OUT_DIR}/")