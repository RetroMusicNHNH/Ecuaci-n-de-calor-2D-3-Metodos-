"""Validación de solucionadores: errores y soluciones analíticas para la ecuación de calor 2D"""

import numpy as np

def solucion_analitica(x, y, t, tipo='senoidal'):
    """Solución analítica para pruebas (placa unidad, sencillas).
    Args:
        x, y: arreglos 1D de coordenadas
        t: tiempo
        tipo: 'senoidal', 'gaussiana', etc.
    Returns:
        u: matriz solución analítica
    """
    X, Y = np.meshgrid(x, y)
    if tipo == 'senoidal':
        return np.exp(-2*np.pi**2*t) * np.sin(np.pi * X) * np.sin(np.pi * Y)
    else:
        raise NotImplementedError('Solo seno disponible.')

def error_l2(u_num, u_exact):
    """Calcula el error relativo L2 entre solución numérica y exacta"""
    return np.linalg.norm(u_num - u_exact) / np.linalg.norm(u_exact)

