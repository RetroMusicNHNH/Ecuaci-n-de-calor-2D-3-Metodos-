"""
Módulo de análisis de estabilidad para métodos de diferencias finitas
Autor: Luis Enrique Reyes García

Incluye:
- Graficar factor de amplificación (Von Neumann) para FTCS
- Verificar condición CFL
- Calcular número de condición de matrices para CN/ADI
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags

# Análisis de Von Neumann para FTCS (factor de amplificación)
def factor_amplificacion_ftcs(alpha, dt, dx, dy, theta_x, theta_y):
    rx = alpha * dt / dx ** 2
    ry = alpha * dt / dy ** 2
    G = 1 - 4 * rx * np.sin(theta_x / 2) ** 2 - 4 * ry * np.sin(theta_y / 2) ** 2
    return G

def graficar_region_estabilidad_ftcs(alpha, dt, dx, dy):
    theta = np.linspace(0, np.pi, 200)
    T1, T2 = np.meshgrid(theta, theta)
    G = factor_amplificacion_ftcs(alpha, dt, dx, dy, T1, T2)
    plt.figure(figsize=(8,6))
    plt.contourf(T1, T2, np.abs(G), levels=[0, 1], cmap='cool')
    plt.xlabel('Theta_x'); plt.ylabel('Theta_y')
    plt.title('Región de estabilidad FTCS: |G| <= 1')
    plt.colorbar(label='|G|')
    plt.tight_layout()
    plt.show()

# Función para verificar CFL FTCS
def verificar_cfl_ftcs(alpha, dx, dy, dt):
    cfl = dt <= 0.25 * (1 / (alpha * (1/(dx**2) + 1/(dy**2))))
    print(f"¿Cumple CFL para FTCS?: {'Sí' if cfl else 'No'}")
    return cfl

# Número de condición de matrices de Crank-Nicolson y ADI
def calcular_numero_condicion(nx, alpha, dt, dx):
    rx = alpha * dt / (2 * dx ** 2)
    diagonales = [np.ones(nx-2) * (1 + 2*rx), np.ones(nx-3)*(-rx), np.ones(nx-3)*(-rx)]
    A = diags(diagonales, [0, -1, 1]).toarray()
    cond = np.linalg.cond(A)
    print(f"Número de condición (nx={nx}): {cond:.2e}")
    return cond

if __name__ == "__main__":
    # Ejemplo de uso rápido
    alpha = 1.0; dt = 0.0002; dx = dy = 1.0/49
    graficar_region_estabilidad_ftcs(alpha, dt, dx, dy)
    verificar_cfl_ftcs(alpha, dx, dy, dt)
    for tam in [20, 40, 60, 80]:
        dx_tmp = 1.0/(tam-1)
        calcular_numero_condicion(tam, alpha, dt, dx_tmp)
