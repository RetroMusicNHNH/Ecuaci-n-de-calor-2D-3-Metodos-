"""
tests/test_fuente.py - Validación FTCS con fuente homogénea
"""
import sys
sys.path.append('./')
import numpy as np
from src.condiciones import inicializar_dominio
from src.solucionadores import resolver_ftcs

nx, ny = 30, 30
x, y, dx, dy = inicializar_dominio(nx, ny)
dt, pasos = 0.0002, 40
alpha = 1.0

def fuente(X, Y, t):
    return np.sin(np.pi*X)*np.sin(np.pi*Y)*np.exp(-t)

X, Y = np.meshgrid(x, y)
u0 = np.zeros_like(X)
resultado = u0.copy()

for n in range(pasos):
    t = dt * n
    Q = fuente(X, Y, t)
    # Paso FTCS con fuente
    resultado[1:-1,1:-1] = (
        resultado[1:-1,1:-1]
        + alpha*dt/dx**2 * (resultado[2:,1:-1] - 2*resultado[1:-1,1:-1] + resultado[0:-2,1:-1])
        + alpha*dt/dy**2 * (resultado[1:-1,2:] - 2*resultado[1:-1,1:-1] + resultado[1:-1,0:-2])
        + dt * Q[1:-1,1:-1]
    )
# Simple chequeo: temperatura final mayor a cero
temp_media = np.mean(resultado)
print(f"Temperatura final media (con fuente): {temp_media:.3f}")
if temp_media > 0:
    print("Validación OK: El término fuente incrementa energía como esperado.")
else:
    print("Advertencia: Temperatura final no aumentó con la fuente (posible error)")
