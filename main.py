"""
main.py - Presentación y comparación de todos los solucionadores de la ecuación de calor 2D
Autor: Luis Enrique Reyes García
Universidad de Costa Rica
Curso MO-0014 Álgebra Lineal Numérica

Este script:
1. Resuelve la ecuación de calor 2D con FTCS, Crank-Nicolson y ADI
2. Calcula y muestra el error L2 frente a solución analítica
3. Grafica resultados numéricos y analíticos
4. Presenta comparación clara entre los métodos
"""

import numpy as np
import matplotlib.pyplot as plt
from src.condiciones import inicializar_dominio, temperatura_inicial
from src.solucionadores import resolver_ftcs, resolver_cn, resolver_adi
from src.validacion import solucion_analitica, error_l2

# Parámetros generales
nx, ny = 50, 50
lx, ly = 1.0, 1.0
alpha = 1.0
x, y, dx, dy = inicializar_dominio(nx, ny, lx, ly)
tipo_ini = 'senoidal'
tu0 = temperatura_inicial(x, y, tipo=tipo_ini)
dt_ftcs, pasos_ftcs = 0.0002, 50
dt_cn, pasos_cn = 0.002, 25
dt_adi, pasos_adi = 0.004, 15

# Solución analítica final
u_analitica = solucion_analitica(x, y, dt_ftcs * pasos_ftcs, tipo=tipo_ini)

print("\n== FTCS Explícito ==")
sol_ftcs = resolver_ftcs(tu0, dx, dy, dt_ftcs, pasos_ftcs)
err_ftcs = error_l2(sol_ftcs[-1], u_analitica)
print(f"Error L2 FTCS: {err_ftcs:.3e}")

print("\n== Crank-Nicolson ==")
sol_cn = resolver_cn(tu0, dx, dy, dt_cn, pasos_cn)
err_cn = error_l2(sol_cn[-1], u_analitica)
print(f"Error L2 Crank-Nicolson: {err_cn:.3e}")

print("\n== ADI ==")
sol_adi = resolver_adi(tu0, dx, dy, dt_adi, pasos_adi)
err_adi = error_l2(sol_adi[-1], u_analitica)
print(f"Error L2 ADI: {err_adi:.3e}")

# Comparación gráfica
plt.figure(figsize=(13,7))
plt.subplot(2,2,1)
plt.title("FTCS Numérico")
plt.imshow(sol_ftcs[-1], origin='lower', cmap='hot', extent=[0,lx,0,ly])
plt.colorbar()
plt.subplot(2,2,2)
plt.title("Crank-Nicolson Numérico")
plt.imshow(sol_cn[-1], origin='lower', cmap='hot', extent=[0,lx,0,ly])
plt.colorbar()
plt.subplot(2,2,3)
plt.title("ADI Numérico")
plt.imshow(sol_adi[-1], origin='lower', cmap='hot', extent=[0,lx,0,ly])
plt.colorbar()
plt.subplot(2,2,4)
plt.title("Solución Analítica")
plt.imshow(u_analitica, origin='lower', cmap='hot', extent=[0,lx,0,ly])
plt.colorbar()
plt.tight_layout()
plt.show()

# Resumen comparativo
print("\n=== Resumen de errores L2 ===")
print(f"FTCS: {err_ftcs:.3e}")
print(f"Crank-Nicolson: {err_cn:.3e}")
print(f"ADI: {err_adi:.3e}")
