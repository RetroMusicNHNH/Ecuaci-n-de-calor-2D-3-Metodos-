"""
tests/test_neumann.py - Validación Neumann (conservación de energía)
"""
import sys
sys.path.append('./')
import numpy as np
from src.condiciones import inicializar_dominio, temperatura_inicial, aplicar_frontera_neumann
from src.solucionadores import resolver_ftcs

nx, ny = 30, 30
x, y, dx, dy = inicializar_dominio(nx, ny)
u0 = np.ones((nx, ny)) # temperatura uniforme

def energia(u, dx, dy):
    return np.sum(u) * dx * dy

sols = resolver_ftcs(u0, dx, dy, dt=0.0002, pasos=30, tipo_frontera='neumann', valor_frontera=0.0)
energias = [energia(u, dx, dy) for u in sols]
print("Energía inicial:", energias[0])
print("Energía final:", energias[-1])
print("Error relativo:", abs(energias[-1] - energias[0]) / energias[0])
if abs(energias[-1] - energias[0]) / energias[0] < 1e-6:
    print("Validación OK: Conservación de energía con Neumann")
else:
    print("Advertencia: No se conserva la energía (posible error)")
