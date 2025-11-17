"""Ejemplo de uso: FTCS para la ecuación de calor 2D"""
import sys
sys.path.append('./')
import numpy as np
import matplotlib.pyplot as plt
from src.condiciones import inicializar_dominio, temperatura_inicial, aplicar_frontera_dirichlet
from src.solucionadores import resolver_ftcs
from src.validacion import solucion_analitica, error_l2

# Parámetros
nx, ny = 50, 50
lx, ly = 1.0, 1.0
dt = 0.0002
pasos = 50
alpha = 1.0

# Crear malla y condiciones iniciales
x, y, dx, dy = inicializar_dominio(nx, ny, lx, ly)
u0 = temperatura_inicial(x, y, tipo='senoidal')

# Resolver con FTCS
soluciones = resolver_ftcs(u0, dx, dy, dt, pasos, alpha, tipo_frontera='dirichlet', valor_frontera=0.0)

# Validar con solución exacta
u_exact = solucion_analitica(x, y, dt*pasos, tipo='senoidal')
err = error_l2(soluciones[-1], u_exact)
print(f"Error L2 final: {err:.6e}")

# Graficar
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.title("FTCS Numérico")
plt.imshow(soluciones[-1], origin='lower', cmap='hot', extent=[0,lx,0,ly])
plt.colorbar()
plt.subplot(1,2,2)
plt.title("Solución Analítica")
plt.imshow(u_exact, origin='lower', cmap='hot', extent=[0,lx,0,ly])
plt.colorbar()
plt.tight_layout()
plt.show()
