import deepxde as dde
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Создаём папку для выходных файлов
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"📁 Создана папка: {output_dir}")

# Создаём подпапку с временной меткой для каждого запуска
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = os.path.join(output_dir, f"run_{timestamp}")
os.makedirs(run_dir)
print(f"📁 Результаты будут сохранены в: {run_dir}")

# Геометрия
geom = dde.geometry.Interval(0, 1)

# Граничные условия
def boundary_left(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)

def boundary_right(x, on_boundary):
    return on_boundary and np.isclose(x[0], 1)

bc_left = dde.icbc.DirichletBC(geom, lambda x: 0.0, boundary_left)
bc_right = dde.icbc.DirichletBC(geom, lambda x: 0.0, boundary_right)

# Правая часть уравнения
def source(x):
    if hasattr(x, 'shape') and hasattr(x, 'dtype') and 'float' in str(x.dtype):
        return -tf.constant(np.pi**2, dtype=tf.float32) * tf.sin(tf.constant(np.pi, dtype=tf.float32) * x)
    else:
        return -np.pi**2 * np.sin(np.pi * x)

# Уравнение Пуассона: u_xx = f(x)
def pde(x, u):
    du_xx = dde.grad.hessian(u, x)
    return du_xx - source(x)

# Создаем данные
data = dde.data.PDE(
    geom,
    pde,
    [bc_left, bc_right],
    num_domain=100,
    num_boundary=2,
    solution=lambda x: np.sin(np.pi * x),
    num_test=100
)

# Улучшенная нейросеть
layer_size = [1] + [100] * 4 + [1]
net = dde.nn.FNN(layer_size, "tanh", "Glorot uniform")

# Модель
model = dde.Model(data, net)

# Компиляция
model.compile("adam", lr=0.001, loss="MSE", metrics=["l2 relative error"])

# Изменяем путь для сохранения модели
checkpoint_path = os.path.join(run_dir, "model_checkpoint")
callbacks = [
    dde.callbacks.EarlyStopping(min_delta=1e-6, patience=2000),
    dde.callbacks.ModelCheckpoint(checkpoint_path, save_better_only=True)
]

# Обучение
print("Начинаем обучение...")
losshistory, train_state = model.train(
    iterations=10000, 
    display_every=1000,
    callbacks=callbacks
)

# Сохраняем историю обучения в файл
loss_history_path = os.path.join(run_dir, "loss_history.txt")
with open(loss_history_path, 'w') as f:
    f.write("Step\tTrain Loss\tTest Loss\tTest Metric\n")
    # Определяем количество шагов
    n_steps = len(losshistory.loss_train)
    for i in range(n_steps):
        step = i * 1000 if i < n_steps else (n_steps - 1) * 1000
        train_loss = losshistory.loss_train[i]
        test_loss = losshistory.loss_test[i] if i < len(losshistory.loss_test) else losshistory.loss_test[-1]
        test_metric = losshistory.metrics_test[i] if i < len(losshistory.metrics_test) else []
        f.write(f"{step}\t{train_loss}\t{test_loss}\t{test_metric}\n")

print(f"💾 История потерь сохранена в: {loss_history_path}")

# Визуализация истории обучения (ручная)
fig_loss, ax_loss = plt.subplots(figsize=(10, 6))
steps = [i * 1000 for i in range(len(losshistory.loss_train))]
ax_loss.semilogy(steps, losshistory.loss_train, 'b-', label='Train Loss', linewidth=2)
ax_loss.semilogy(steps, losshistory.loss_test, 'r-', label='Test Loss', linewidth=2)
ax_loss.set_xlabel('Iterations')
ax_loss.set_ylabel('Loss')
ax_loss.set_title('History of Loss')
ax_loss.legend()
ax_loss.grid(True)
loss_plot_path = os.path.join(run_dir, "loss_plot.png")
plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
print(f"💾 График потерь сохранён в: {loss_plot_path}")
plt.close(fig_loss)

# Сравнение с точным решением
x_test = np.linspace(0, 1, 200)[:, None]
y_pred = model.predict(x_test)
y_exact = np.sin(np.pi * x_test)

# Подробная визуализация
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# График 1: Сравнение решений
axes[0, 0].plot(x_test, y_exact, 'b-', label='Точное решение', linewidth=2)
axes[0, 0].plot(x_test, y_pred, 'r--', label='PINN решение', linewidth=2)
axes[0, 0].set_xlabel('x')
axes[0, 0].set_ylabel('u(x)')
axes[0, 0].legend()
axes[0, 0].set_title('Сравнение решений')
axes[0, 0].grid(True)

# График 2: Абсолютная ошибка
error_abs = np.abs(y_pred - y_exact)
axes[0, 1].semilogy(x_test, error_abs, 'g-', linewidth=2)
axes[0, 1].set_xlabel('x')
axes[0, 1].set_ylabel('|u_pred - u_exact|')
axes[0, 1].set_title('Абсолютная ошибка (логарифмическая шкала)')
axes[0, 1].grid(True)

# График 3: Относительная ошибка
error_rel = error_abs / (np.abs(y_exact) + 1e-8)
axes[1, 0].plot(x_test, error_rel * 100, 'm-', linewidth=2)
axes[1, 0].set_xlabel('x')
axes[1, 0].set_ylabel('Относительная ошибка (%)')
axes[1, 0].set_title('Относительная ошибка')
axes[1, 0].grid(True)

# График 4: Градиент решения
du_exact = np.pi * np.cos(np.pi * x_test)
du_pred = np.gradient(y_pred.ravel(), x_test.ravel())[:, None]
axes[1, 1].plot(x_test, du_exact, 'b-', label='Точный градиент', linewidth=2)
axes[1, 1].plot(x_test, du_pred, 'r--', label='PINN градиент', linewidth=2)
axes[1, 1].set_xlabel('x')
axes[1, 1].set_ylabel("u'(x)")
axes[1, 1].legend()
axes[1, 1].set_title('Производная решения')
axes[1, 1].grid(True)

plt.tight_layout()

# Сохраняем график
plot_path = os.path.join(run_dir, "solution_comparison.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"💾 График сохранён в: {plot_path}")
plt.show()

# Вычисление ошибок
l2_error = np.linalg.norm(y_pred - y_exact) / np.linalg.norm(y_exact)
max_error = np.max(np.abs(y_pred - y_exact))
mean_error = np.mean(np.abs(y_pred - y_exact))


# Сохраняем предсказания в CSV файл
csv_path = os.path.join(run_dir, "predictions.csv")
np.savetxt(
    csv_path, 
    np.column_stack([x_test.flatten(), y_pred.flatten(), y_exact.flatten(), error_abs.flatten()]),
    header="x, u_pred, u_exact, absolute_error",
    delimiter=",",
    comments=""
)
print(f"💾 Предсказания сохранены в: {csv_path}")

# Сохраняем параметры модели
params_path = os.path.join(run_dir, "model_params.txt")
with open(params_path, 'w', encoding='utf-8') as f:
    f.write("ПАРАМЕТРЫ МОДЕЛИ:\n")
    f.write("="*50 + "\n")
    f.write(f"Архитектура сети: {layer_size}\n")
    f.write(f"Функция активации: tanh\n")
    f.write(f"Инициализатор: Glorot uniform\n")
    f.write(f"Оптимизатор: Adam\n")
    f.write(f"Скорость обучения: 0.001\n")
    f.write(f"Количество точек внутри области: 100\n")
    f.write(f"Количество точек на границе: 2\n")
    f.write(f"Количество тестовых точек: 100\n")
    f.write(f"Early stopping patience: 2000\n")
    f.write(f"min_delta: 1e-6\n")

print(f"💾 Параметры модели сохранены в: {params_path}")

print("\n" + "="*50)
print("РЕЗУЛЬТАТЫ ОБУЧЕНИЯ:")
print("="*50)
print(f'✅ Относительная L2 ошибка: {l2_error:.6f} ({l2_error*100:.4f}%)')
print(f'✅ Максимальная абсолютная ошибка: {max_error:.6f}')
print(f'✅ Средняя абсолютная ошибка: {mean_error:.6f}')
print("="*50)
print(f"\n📁 Все файлы сохранены в папке: {run_dir}")
print("📁 Содержимое папки:")
for file in os.listdir(run_dir):
    file_size = os.path.getsize(os.path.join(run_dir, file))
    if file_size < 1024:
        size_str = f"{file_size} B"
    elif file_size < 1024*1024:
        size_str = f"{file_size/1024:.1f} KB"
    else:
        size_str = f"{file_size/(1024*1024):.1f} MB"
    print(f"   - {file} ({size_str})")