"""Condiciones de frontera e iniciales para ecuación de calor 2D"""

import numpy as np


def inicializar_dominio(nx, ny, lx=1.0, ly=1.0):
    """Crea malla espacial
    
    Args:
        nx: Puntos en x
        ny: Puntos en y
        lx: Longitud en x
        ly: Longitud en y
    
    Returns:
        x, y, dx, dy: Arreglos espaciales y pasos
    """
    x = np.linspace(0, lx, nx)
    y = np.linspace(0, ly, ny)
    dx = lx / (nx - 1)
    dy = ly / (ny - 1)
    return x, y, dx, dy


def temperatura_inicial(x, y, tipo='cero'):
    """Define distribución inicial de temperatura
    
    Args:
        x, y: Arreglos de coordenadas
        tipo: 'cero', 'gaussiana', 'senoidal'
    
    Returns:
        u0: Matriz de temperatura inicial
    """
    X, Y = np.meshgrid(x, y)
    
    if tipo == 'cero':
        return np.zeros_like(X)
    
    elif tipo == 'gaussiana':
        cx, cy = 0.5, 0.5  # Centro
        sigma = 0.1
        return np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))
    
    elif tipo == 'senoidal':
        return np.sin(np.pi * X) * np.sin(np.pi * Y)
    
    else:
        raise ValueError(f"Tipo '{tipo}' no reconocido")


def aplicar_frontera_dirichlet(u, valor=0.0):
    """Aplica condición de Dirichlet (temperatura fija)
    
    Args:
        u: Matriz de temperatura
        valor: Temperatura en frontera
    """
    u[0, :] = valor   # Borde inferior
    u[-1, :] = valor  # Borde superior
    u[:, 0] = valor   # Borde izquierdo
    u[:, -1] = valor  # Borde derecho


def aplicar_frontera_neumann(u, dx, dy, flujo=0.0):
    """Aplica condición de Neumann (flujo fijo)
    
    Args:
        u: Matriz de temperatura
        dx, dy: Pasos espaciales
        flujo: Flujo en frontera (derivada)
    """
    # Aproximación de segundo orden
    u[0, :] = u[1, :] - flujo * dy      # Borde inferior
    u[-1, :] = u[-2, :] + flujo * dy    # Borde superior
    u[:, 0] = u[:, 1] - flujo * dx      # Borde izquierdo
    u[:, -1] = u[:, -2] + flujo * dx    # Borde derecho


def aplicar_frontera_mixta(u, dx, dy, borde_dirichlet, borde_neumann):
    """Aplica combinación de Dirichlet y Neumann
    
    Args:
        u: Matriz de temperatura
        dx, dy: Pasos espaciales
        borde_dirichlet: dict con bordes y valores {'inferior': valor, ...}
        borde_neumann: dict con bordes y flujos {'superior': flujo, ...}
    """
    # Dirichlet
    for borde, valor in borde_dirichlet.items():
        if borde == 'inferior':
            u[0, :] = valor
        elif borde == 'superior':
            u[-1, :] = valor
        elif borde == 'izquierdo':
            u[:, 0] = valor
        elif borde == 'derecho':
            u[:, -1] = valor
    
    # Neumann
    for borde, flujo in borde_neumann.items():
        if borde == 'inferior':
            u[0, :] = u[1, :] - flujo * dy
        elif borde == 'superior':
            u[-1, :] = u[-2, :] + flujo * dy
        elif borde == 'izquierdo':
            u[:, 0] = u[:, 1] - flujo * dx
        elif borde == 'derecho':
            u[:, -1] = u[:, -2] + flujo * dx
