"""
utils/convergencia.py - Tabla L2 para validación de orden de convergencia
"""
import sys
sys.path.append('./')
import numpy as np
from src.condiciones import inicializar_dominio, temperatura_inicial
from src.solucionadores import resolver_ftcs
from src.validacion import solucion_analitica, error_l2

h_list = [0.05, 0.025, 0.0125]
print("h\tError_L2")

for h in h_list:
    nx = int(1/h) + 1
    x, y, dx, dy = inicializar_dominio(nx, nx, lx=1.0, ly=1.0)
    u0 = temperatura_inicial(x, y, tipo='senoidal')
    dt = 0.2 * h**2
    pasos = 10
    soluciones = resolver_ftcs(u0, dx, dy, dt, pasos)
    u_exact = solucion_analitica(x, y, dt*pasos, tipo='senoidal')
    err = error_l2(soluciones[-1], u_exact)
    print(f"{h:.4f}\t{err:.2e}")
