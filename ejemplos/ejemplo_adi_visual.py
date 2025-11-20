"""Ejemplo visual ADI para ecuación de calor 2D"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from src.solucionadores import resolver_adi

# Parámetros de la malla y físicos
N = 20  # Tamaño de la malla (NxN)
Lx, Ly = 1.0, 1.0
alpha = 1.0

dx = Lx / (N - 1)
dy = Ly / (N - 1)
dt = 0.9 * dx**2 / (4 * alpha) * 4   # ADI permite dt más grande
T_final = 0.1
pasos = int(T_final / dt)

# Crear malla inicial
x = np.linspace(0, Lx, N)
y = np.linspace(0, Ly, N)
X, Y = np.meshgrid(x, y)
u0 = np.sin(np.pi * X) * np.sin(np.pi * Y)

# Ejecutar resolución
sols = resolver_adi(u0, dx, dy, dt, pasos, alpha)

# --- Visualización de malla ---
fig, ax = plt.subplots(figsize=(5, 4))
ax.set_title(f'Malla 2D ({N}x{N}) y condición inicial')
ax.plot(X, Y, marker='o', linestyle='none', color='gray', markersize=3, alpha=0.6)
heat = ax.contourf(X, Y, u0, levels=30, cmap='plasma')
cbar = fig.colorbar(heat, ax=ax)
cbar.set_label('Temperatura inicial')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.tight_layout()
plt.show()

# --- Animación de la solución ---
fig2, ax2 = plt.subplots(figsize=(5.8, 4.5))
grafico = ax2.imshow(sols[0], cmap='plasma', origin='lower', extent=(0, Lx, 0, Ly), vmin=0, vmax=1)
cbar2 = fig2.colorbar(grafico, ax=ax2)
cbar2.set_label('Temperatura')
txt = ax2.text(0.05, 1.03, '', transform=ax2.transAxes)

ax2.set_title('Evolución de temperatura (ADI)')
ax2.set_xlabel('x')
ax2.set_ylabel('y')


def animar(i):
    grafico.set_array(sols[i])
    txt.set_text(f'Paso {i}/{pasos} | t={i * dt:.3f} s')
    return grafico, txt

anim = animation.FuncAnimation(
    fig2, animar, frames=range(0, pasos+1, max(1, pasos//80)), interval=60, blit=False)
plt.show()
