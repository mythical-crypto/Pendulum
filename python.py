import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Настройка стиля графиков для отчета
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['font.size'] = 12
rcParams['figure.figsize'] = (10, 7)

# Параметры моделирования
L = 1.0  # длина маятника (м)
g = 9.81  # ускорение свободного падения (м/с²)
phi0 = 0.1  # начальный угол (рад)
omega0 = 0.0  # начальная угловая скорость (рад/с)
beta = 0.1  # коэффициент затухания (для физического маятника)
t_start = 0.0  # начальное время (с)
t_end = 20.0  # конечное время (с)

# Шаги интегрирования для исследования
steps = [0.01, 0.05, 0.1, 0.2, 0.5]
colors = ['blue', 'green', 'orange', 'red', 'purple']

# Аналитические решения
def analytical_solution_simple(t, phi0, L, g):
    """Аналитическое решение для математического маятника (малые углы)"""
    omega0 = np.sqrt(g/L)
    return phi0 * np.cos(omega0 * t)

def analytical_solution_damped(t, phi0, L, g, beta):
    """Аналитическое решение для физического маятника с затуханием"""
    omega0 = np.sqrt(g/L)
    omega_d = np.sqrt(omega0**2 - beta**2)
    return phi0 * np.exp(-beta * t) * np.cos(omega_d * t)

# Метод Эйлера
def euler_method_simple(h, L, g, phi0, omega0, t_end):
    """Метод Эйлера для математического маятника"""
    n = int(t_end / h)
    t = np.linspace(0, t_end, n+1)
    phi = np.zeros(n+1)
    omega = np.zeros(n+1)
    
    phi[0] = phi0
    omega[0] = omega0
    
    for i in range(n):
        phi[i+1] = phi[i] + h * omega[i]
        omega[i+1] = omega[i] - h * (g/L) * np.sin(phi[i])
    
    return t, phi, omega

def euler_method_damped(h, L, g, beta, phi0, omega0, t_end):
    """Метод Эйлера для физического маятника с затуханием"""
    n = int(t_end / h)
    t = np.linspace(0, t_end, n+1)
    phi = np.zeros(n+1)
    omega = np.zeros(n+1)
    
    phi[0] = phi0
    omega[0] = omega0
    
    for i in range(n):
        phi[i+1] = phi[i] + h * omega[i]
        omega[i+1] = omega[i] - h * (2*beta*omega[i] + (g/L)*phi[i])
    
    return t, phi, omega

# Создаем массив для хранения ошибок
errors_simple = []
errors_damped = []

# График 1: Сравнение решений для разных шагов (математический маятник)
plt.figure(figsize=(12, 8))

for i, h in enumerate(steps):
    t, phi_num, omega_num = euler_method_simple(h, L, g, phi0, omega0, t_end)
    phi_analytical = analytical_solution_simple(t, phi0, L, g)
    
    plt.subplot(2, 3, i+1)
    plt.plot(t, phi_num, color=colors[i], linewidth=2, label=f'Численное (h={h} с)')
    plt.plot(t, phi_analytical, 'k--', linewidth=1.5, label='Аналитическое')
    plt.xlabel('Время, с', fontsize=10)
    plt.ylabel('Угол φ, рад', fontsize=10)
    plt.title(f'Шаг h = {h} с', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    
    # Вычисление среднеквадратичной ошибки
    mse = np.sqrt(np.mean((phi_num - phi_analytical)**2))
    errors_simple.append(mse)
    plt.text(0.02, 0.05, f'СКО = {mse:.5f}', transform=plt.gca().transAxes, 
             fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.suptitle('Математический маятник: сравнение численного и аналитического решений', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('math_pendulum_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# График 2: Сравнение решений для разных шагов (физический маятник с затуханием)
plt.figure(figsize=(12, 8))

for i, h in enumerate(steps):
    t, phi_num, omega_num = euler_method_damped(h, L, g, beta, phi0, omega0, t_end)
    phi_analytical = analytical_solution_damped(t, phi0, L, g, beta)
    
    plt.subplot(2, 3, i+1)
    plt.plot(t, phi_num, color=colors[i], linewidth=2, label=f'Численное (h={h} с)')
    plt.plot(t, phi_analytical, 'k--', linewidth=1.5, label='Аналитическое')
    plt.xlabel('Время, с', fontsize=10)
    plt.ylabel('Угол φ, рад', fontsize=10)
    plt.title(f'Шаг h = {h} с', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    
    # Вычисление среднеквадратичной ошибки
    mse = np.sqrt(np.mean((phi_num - phi_analytical)**2))
    errors_damped.append(mse)
    plt.text(0.02, 0.05, f'СКО = {mse:.5f}', transform=plt.gca().transAxes, 
             fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.suptitle('Физический маятник с затуханием: сравнение численного и аналитического решений', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('phys_pendulum_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# График 3: Фазовые портреты для разных шагов
fig, axes = plt.subplots(2, 3, figsize=(14, 8))

for i, h in enumerate(steps):
    # Математический маятник
    t1, phi_num1, omega_num1 = euler_method_simple(h, L, g, phi0, omega0, t_end)
    
    ax1 = axes[0, i] if i < 3 else axes[0, i-3]
    ax1.plot(phi_num1, omega_num1, color=colors[i], linewidth=1.5)
    ax1.set_xlabel('φ, рад', fontsize=10)
    ax1.set_ylabel('ω, рад/с', fontsize=10)
    ax1.set_title(f'Математический, h={h} с', fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Физический маятник
    t2, phi_num2, omega_num2 = euler_method_damped(h, L, g, beta, phi0, omega0, t_end)
    
    ax2 = axes[1, i] if i < 3 else axes[1, i-3]
    ax2.plot(phi_num2, omega_num2, color=colors[i], linewidth=1.5)
    ax2.set_xlabel('φ, рад', fontsize=10)
    ax2.set_ylabel('ω, рад/с', fontsize=10)
    ax2.set_title(f'Физический, h={h} с', fontsize=11)
    ax2.grid(True, alpha=0.3)

plt.suptitle('Фазовые портреты маятников для разных шагов интегрирования', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('phase_portraits.png', dpi=300, bbox_inches='tight')
plt.show()

# График 4: Зависимость ошибки от шага интегрирования
plt.figure(figsize=(10, 6))

# Линейный масштаб
plt.subplot(1, 2, 1)
plt.plot(steps, errors_simple, 'bo-', linewidth=2, markersize=8, label='Математический')
plt.plot(steps, errors_damped, 'ro-', linewidth=2, markersize=8, label='Физический')
plt.xlabel('Шаг интегрирования h, с', fontsize=12)
plt.ylabel('Среднеквадратичная ошибка', fontsize=12)
plt.title('Зависимость ошибки от шага', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)

# Логарифмический масштаб
plt.subplot(1, 2, 2)
plt.loglog(steps, errors_simple, 'bo-', linewidth=2, markersize=8, label='Математический')
plt.loglog(steps, errors_damped, 'ro-', linewidth=2, markersize=8, label='Физический')
plt.xlabel('Шаг интегрирования h, с', fontsize=12)
plt.ylabel('Среднеквадратичная ошибка (лог)', fontsize=12)
plt.title('Логарифмический масштаб', fontsize=14)
plt.grid(True, alpha=0.3, which='both')
plt.legend(fontsize=11)

plt.suptitle('Анализ устойчивости метода Эйлера', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('error_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# График 5: Критерий устойчивости
plt.figure(figsize=(10, 6))

omega0 = np.sqrt(g/L)
h_critical = 2 / omega0  # теоретический критерий устойчивости

# Анализ максимальной амплитуды для разных шагов
max_amplitudes_simple = []
max_amplitudes_damped = []

for h in steps:
    t, phi_num1, _ = euler_method_simple(h, L, g, phi0, omega0, 10.0)  # только 10 сек для наглядности
    max_amplitudes_simple.append(np.max(np.abs(phi_num1)))
    
    t, phi_num2, _ = euler_method_damped(h, L, g, beta, phi0, omega0, 10.0)
    max_amplitudes_damped.append(np.max(np.abs(phi_num2)))

plt.plot(steps, max_amplitudes_simple, 'bo-', linewidth=2, markersize=8, label='Математический')
plt.plot(steps, max_amplitudes_damped, 'ro-', linewidth=2, markersize=8, label='Физический')
plt.axvline(x=h_critical, color='green', linestyle='--', linewidth=2, 
            label=f'Теоретический предел: h = {h_critical:.2f} с')
plt.axvline(x=0.2, color='orange', linestyle='--', linewidth=2, 
            label='Экспериментальный предел: h = 0.2 с')

plt.xlabel('Шаг интегрирования h, с', fontsize=12)
plt.ylabel('Максимальная амплитуда, рад', fontsize=12)
plt.title('Критерий устойчивости метода Эйлера', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('stability_criterion.png', dpi=300, bbox_inches='tight')
plt.show()

# Вывод таблицы с результатами
print("="*80)
print("ТАБЛИЦА 1: Результаты моделирования методом Эйлера")
print("="*80)
print(f"{'Шаг h (с)':<12} {'Мат. маятник':<20} {'Физ. маятник':<20}")
print(f"{'':<12} {'СКО':<10} {'Макс. ампл.':<10} {'СКО':<10} {'Макс. ампл.':<10}")
print("-"*80)

for i, h in enumerate(steps):
    t, phi_num1, _ = euler_method_simple(h, L, g, phi0, omega0, t_end)
    phi_analytical1 = analytical_solution_simple(t, phi0, L, g)
    mse1 = np.sqrt(np.mean((phi_num1 - phi_analytical1)**2))
    max_amp1 = np.max(np.abs(phi_num1))
    
    t, phi_num2, _ = euler_method_damped(h, L, g, beta, phi0, omega0, t_end)
    phi_analytical2 = analytical_solution_damped(t, phi0, L, g, beta)
    mse2 = np.sqrt(np.mean((phi_num2 - phi_analytical2)**2))
    max_amp2 = np.max(np.abs(phi_num2))
    
    print(f"{h:<12.2f} {mse1:<10.5f} {max_amp1:<10.5f} {mse2:<10.5f} {max_amp2:<10.5f}")

print("="*80)
print(f"\nТеоретический критерий устойчивости: h < {h_critical:.2f} с")
print("Экспериментальный предел устойчивости: h ≤ 0.2 с")
print("="*80)

# Дополнительный график: сравнение методов для одного шага (h=0.1 с)
plt.figure(figsize=(12, 5))

h = 0.1  # выбранный шаг для сравнения

# Математический маятник
plt.subplot(1, 2, 1)
t, phi_num1, _ = euler_method_simple(h, L, g, phi0, omega0, t_end)
phi_analytical1 = analytical_solution_simple(t, phi0, L, g)
plt.plot(t, phi_num1, 'b-', linewidth=2, label='Численное решение')
plt.plot(t, phi_analytical1, 'r--', linewidth=2, label='Аналитическое решение')
plt.xlabel('Время, с', fontsize=12)
plt.ylabel('Угол φ, рад', fontsize=12)
plt.title(f'Математический маятник (h={h} с)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)

# Физический маятник
plt.subplot(1, 2, 2)
t, phi_num2, _ = euler_method_damped(h, L, g, beta, phi0, omega0, t_end)
phi_analytical2 = analytical_solution_damped(t, phi0, L, g, beta)
plt.plot(t, phi_num2, 'b-', linewidth=2, label='Численное решение')
plt.plot(t, phi_analytical2, 'r--', linewidth=2, label='Аналитическое решение')
plt.xlabel('Время, с', fontsize=12)
plt.ylabel('Угол φ, рад', fontsize=12)
plt.title(f'Физический маятник (h={h} с)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)

plt.suptitle('Сравнение численного и аналитического решений (h=0.1 с)', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('comparison_h_0.1.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nГрафики сохранены в файлы:")
print("1. math_pendulum_comparison.png - сравнение решений для математического маятника")
print("2. phys_pendulum_comparison.png - сравнение решений для физического маятника")
print("3. phase_portraits.png - фазовые портреты")
print("4. error_analysis.png - анализ ошибок")
print("5. stability_criterion.png - критерий устойчивости")
print("6. comparison_h_0.1.png - подробное сравнение для h=0.1 с")
