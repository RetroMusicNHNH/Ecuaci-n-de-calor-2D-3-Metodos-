"""
tests/test_fuente.py - Validación FTCS con fuente con gráfico
Ejemplo: Placa unidad, fuente $Q = \sin(\pi x)\sin(\pi y) e^{-t}$
"""
import sys
sys.path.append('./')
import numpy as np
import matplotlib.pyplot as plt
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
hist = []

for n in range(pasos):
    t = dt * n
    Q = fuente(X, Y, t)
    resultado[1:-1,1:-1] = (
        resultado[1:-1,1:-1]
        + alpha*dt/dx**2 * (resultado[2:,1:-1] - 2*resultado[1:-1,1:-1] + resultado[0:-2,1:-1])
        + alpha*dt/dy**2 * (resultado[1:-1,2:] - 2*resultado[1:-1,1:-1] + resultado[1:-1,0:-2])
        + dt * Q[1:-1,1:-1]
    )
    hist.append(np.mean(resultado))

# Mostrar evolución y matriz final
print(f"Temperatura final media (con fuente): {np.mean(resultado):.3f}")
if np.mean(resultado) > 0:
    print("Validación OK: El término fuente incrementa energía como esperado.")
else:
    print("Advertencia: Temperatura final no aumentó con la fuente (posible error)")

plt.plot(hist)
plt.title("Evolución de la temperatura media (fuente)")
plt.xlabel("Paso temporal")
plt.ylabel("Temperatura media")
plt.grid()
plt.tight_layout()
plt.show()

plt.imshow(resultado, origin='lower', cmap='inferno')
plt.colorbar()
plt.title("Distribución final de temperatura (fuente)")
plt.show()
