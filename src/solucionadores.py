"""Solucionadores para la ecuación de calor 2D: FTCS, Crank-Nicolson y ADI (corregido)"""

import numpy as np
from src.condiciones import aplicar_frontera_dirichlet, aplicar_frontera_neumann

def resolver_ftcs(u0, dx, dy, dt, pasos, alpha=1.0, tipo_frontera='dirichlet', valor_frontera=0.0):
    """Resuelve la ecuación de calor 2D usando FTCS explícito."""
    u = u0.copy()
    nx, ny = u.shape
    r_x = alpha * dt / dx**2
    r_y = alpha * dt / dy**2
    soluciones = [u.copy()]
    for n in range(pasos):
        u_new = u.copy()
        u_new[1:-1,1:-1] = (
            u[1:-1,1:-1] +
            r_x * (u[2:,1:-1] - 2*u[1:-1,1:-1] + u[0:-2,1:-1]) +
            r_y * (u[1:-1,2:] - 2*u[1:-1,1:-1] + u[1:-1,0:-2])
        )
        if tipo_frontera == 'dirichlet':
            aplicar_frontera_dirichlet(u_new, valor_frontera)
        elif tipo_frontera == 'neumann':
            aplicar_frontera_neumann(u_new, dx, dy, valor_frontera)
        else:
            raise ValueError('Tipo de frontera no soportado.')
        u = u_new
        soluciones.append(u.copy())
    return soluciones

def resolver_cn(u0, dx, dy, dt, pasos, alpha=1.0, tipo_frontera='dirichlet', valor_frontera=0.0):
    """Resuelve la ecuación de calor 2D usando Crank-Nicolson implícito (simple, para mallas pequeñas)."""
    from scipy.sparse import diags
    from scipy.sparse.linalg import spsolve

    u = u0.copy()
    nx, ny = u.shape
    r_x = alpha * dt / (2*dx**2)
    r_y = alpha * dt / (2*dy**2)
    soluciones = [u.copy()]
    for n in range(pasos):
        u_med = u.copy()
        for j in range(1, ny-1):
            A = diags([-r_x, 1+2*r_x, -r_x], [-1,0,1], shape=(nx-2, nx-2)).tocsc()
            b = (
                r_y * u[1:-1, j+1] +
                (1-2*r_y)*u[1:-1, j] +
                r_y * u[1:-1, j-1]
            )
            u_med[1:-1, j] = spsolve(A, b)
        u_new = u_med.copy()
        for i in range(1, nx-1):
            A = diags([-r_y, 1+2*r_y, -r_y], [-1,0,1], shape=(ny-2, ny-2)).tocsc()
            b = (
                r_x * u_med[i+1, 1:-1] +
                (1-2*r_x)*u_med[i, 1:-1] +
                r_x * u_med[i-1, 1:-1]
            )
            u_new[i, 1:-1] = spsolve(A, b)
        if tipo_frontera == 'dirichlet':
            aplicar_frontera_dirichlet(u_new, valor_frontera)
        elif tipo_frontera == 'neumann':
            aplicar_frontera_neumann(u_new, dx, dy, valor_frontera)
        else:
            raise ValueError('Tipo de frontera no soportado.')
        u = u_new
        soluciones.append(u.copy())
    return soluciones

def resolver_adi(u0, dx, dy, dt, pasos, alpha=1.0, tipo_frontera='dirichlet', valor_frontera=0.0):
    """Resuelve la ecuación de calor 2D usando el método ADI (Peaceman-Rachford) corregido."""
    from scipy.sparse import diags
    from scipy.sparse.linalg import spsolve

    u = u0.copy()
    nx, ny = u.shape
    r_x = alpha * dt / (2*dx**2)
    r_y = alpha * dt / (2*dy**2)
    soluciones = [u.copy()]
    for n in range(pasos):
        # Paso 1: x-implícito, y-explícito
        u_med = u.copy()
        for j in range(1, ny-1):
            A = diags([-r_x, 1 + 2*r_x, -r_x], [-1,0,1], shape=(nx-2, nx-2)).tocsc()
            b = r_y*(u[1:-1, j+1] - 2*u[1:-1, j] + u[1:-1, j-1]) + u[1:-1, j]
            u_med[1:-1, j] = spsolve(A, b)
        # Paso 2: y-implícito, x-explícito
        u_new = u_med.copy()
        for i in range(1, nx-1):
            A = diags([-r_y, 1 + 2*r_y, -r_y], [-1,0,1], shape=(ny-2, ny-2)).tocsc()
            b = r_x*(u_med[i+1,1:-1] - 2*u_med[i,1:-1] + u_med[i-1,1:-1]) + u_med[i,1:-1]
            u_new[i, 1:-1] = spsolve(A, b)
        if tipo_frontera == 'dirichlet':
            aplicar_frontera_dirichlet(u_new, valor_frontera)
        elif tipo_frontera == 'neumann':
            aplicar_frontera_neumann(u_new, dx, dy, valor_frontera)
        u = u_new
        soluciones.append(u.copy())
    return soluciones
