"""Ejemplo de uso: Crank-Nicolson para ecuación de calor 2D"""
import numpy as np
import matplotlib.pyplot as plt
from src.condiciones import inicializar_dominio, temperatura_inicial
from src.solucionadores import resolver_cn
from src.validacion import solucion_analitica, error_l2

nx, ny = 50, 50
lx, ly = 1.0, 1.0
dt = 0.002
pasos = 25
alpha = 1.0
x, y, dx, dy = inicializar_dominio(nx, ny, lx, ly)
u0 = temperatura_inicial(x, y, tipo='senoidal')
soluciones = resolver_cn(u0, dx, dy, dt, pasos, alpha, 'dirichlet', 0.0)
u_exact = solucion_analitica(x, y, dt*pasos, 'senoidal')
err = error_l2(soluciones[-1], u_exact)
print(f"Error L2 final: {err:.6e}")
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.title("Crank-Nicolson Numérico")
plt.imshow(soluciones[-1], origin='lower', cmap='hot', extent=[0,lx,0,ly])
plt.colorbar()
plt.subplot(1,2,2)
plt.title("Solución Analítica")
plt.imshow(u_exact, origin='lower', cmap='hot', extent=[0,lx,0,ly])
plt.colorbar()
plt.tight_layout()
plt.show()
