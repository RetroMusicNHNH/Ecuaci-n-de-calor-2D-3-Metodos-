"""Test de convergencia: compara el error L2 a diferentes tamaños de malla y paso de tiempo."""
import sys
sys.path.append('./')
import numpy as np
from src.condiciones import inicializar_dominio, temperatura_inicial
from src.solucionadores import resolver_ftcs
from src.validacion import solucion_analitica, error_l2

def test_convergencia_ftcs():
    resultados = []
    for nx in [20, 40, 60, 80]:
        ny = nx
        x, y, dx, dy = inicializar_dominio(nx, ny, lx=1.0, ly=1.0)
        u0 = temperatura_inicial(x, y, tipo='senoidal')
        dt = 0.2 * min(dx, dy)**2 # estabilidad
        pasos = 10
        soluciones = resolver_ftcs(u0, dx, dy, dt, pasos)
        u_exact = solucion_analitica(x, y, dt*pasos, tipo='senoidal')
        err = error_l2(soluciones[-1], u_exact)
        resultados.append((nx, err))
        print(f"nx={nx}, Error L2={err:.2e}")
    return resultados

if __name__ == "__main__":
    test_convergencia_ftcs()
