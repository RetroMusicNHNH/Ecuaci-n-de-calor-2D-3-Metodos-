"""Comparación visual lado a lado: FTCS, Crank-Nicolson y ADI"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.solucionadores import resolver_ftcs, resolver_cn, resolver_adi

# Parámetros de la malla y físicos
N = 20
Lx, Ly = 1.0, 1.0
alpha = 1.0

dx = Lx / (N - 1)
dy = Ly / (N - 1)
dt_ftcs = 0.9 * dx**2 / (4 * alpha)
dt_implicito = 4 * dt_ftcs
T_final = 0.05

pasos_ftcs = int(T_final / dt_ftcs)
pasos_implicito = int(T_final / dt_implicito)

# Condición inicial
x = np.linspace(0, Lx, N)
y = np.linspace(0, Ly, N)
X, Y = np.meshgrid(x, y)
u0 = np.sin(np.pi * X) * np.sin(np.pi * Y)

print("Resolviendo con FTCS...")
sol_ftcs = resolver_ftcs(u0, dx, dy, dt_ftcs, pasos_ftcs, alpha)

print("Resolviendo con Crank-Nicolson...")
sol_cn = resolver_cn(u0, dx, dy, dt_implicito, pasos_implicito, alpha)

print("Resolviendo con ADI...")
sol_adi = resolver_adi(u0, dx, dy, dt_implicito, pasos_implicito, alpha)

# Comparación en tiempo final
fig, axes = plt.subplots(1, 3, figsize=(13, 4))

# FTCS
im1 = axes[0].imshow(sol_ftcs[-1], cmap='inferno', origin='lower', extent=(0, Lx, 0, Ly), vmin=0, vmax=1)
axes[0].set_title(f'FTCS\n{pasos_ftcs} pasos | dt={dt_ftcs:.5f}')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
fig.colorbar(im1, ax=axes[0], label='Temperatura')

# Crank-Nicolson
im2 = axes[1].imshow(sol_cn[-1], cmap='cool', origin='lower', extent=(0, Lx, 0, Ly), vmin=0, vmax=1)
axes[1].set_title(f'Crank-Nicolson\n{pasos_implicito} pasos | dt={dt_implicito:.5f}')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
fig.colorbar(im2, ax=axes[1], label='Temperatura')

# ADI
im3 = axes[2].imshow(sol_adi[-1], cmap='plasma', origin='lower', extent=(0, Lx, 0, Ly), vmin=0, vmax=1)
axes[2].set_title(f'ADI\n{pasos_implicito} pasos | dt={dt_implicito:.5f}')
axes[2].set_xlabel('x')
axes[2].set_ylabel('y')
fig.colorbar(im3, ax=axes[2], label='Temperatura')

plt.suptitle(f'Comparación de Métodos en t={T_final} s', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Comparación de perfil 1D en y=0.5
fig2, ax = plt.subplots(figsize=(7, 4.5))
y_idx = N // 2

ax.plot(x, sol_ftcs[-1][y_idx, :], marker='o', label='FTCS', linewidth=2)
ax.plot(x, sol_cn[-1][y_idx, :], marker='s', label='Crank-Nicolson', linewidth=2)
ax.plot(x, sol_adi[-1][y_idx, :], marker='^', label='ADI', linewidth=2)

ax.set_xlabel('x', fontsize=11)
ax.set_ylabel('Temperatura', fontsize=11)
ax.set_title(f'Perfil de temperatura en y=0.5 (t={T_final} s)', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\nComparación completada")
