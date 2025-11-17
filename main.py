"""
main.py - Presentación y comparación de solucionadores + ejemplo real modelado
Autor: Luis Enrique Reyes García
Universidad de Costa Rica
Curso MO-0014 Álgebra Lineal Numérica

Añade:
- Gráficas individuales de matrices de resultados para FTCS, CN, ADI y analítica
- Modelo real: enfriamiento de una placa con borde superior caliente y resto frío
"""

import numpy as np
import matplotlib.pyplot as plt
from src.condiciones import inicializar_dominio, temperatura_inicial, aplicar_frontera_dirichlet
from src.solucionadores import resolver_ftcs, resolver_cn, resolver_adi
from src.validacion import solucion_analitica, error_l2

#------------------ Comparación problema seno (como antes) ------------------#
x, y, dx, dy = inicializar_dominio(50, 50, lx=1.0, ly=1.0)
tu0 = temperatura_inicial(x, y, tipo='senoidal')
dt_ftcs, pasos_ftcs = 0.0002, 50
dt_cn, pasos_cn = 0.002, 25
dt_adi, pasos_adi = 0.004, 15

u_exact = solucion_analitica(x, y, dt_ftcs*pasos_ftcs, tipo='senoidal')
sol_ftcs = resolver_ftcs(tu0, dx, dy, dt_ftcs, pasos_ftcs)
sol_cn = resolver_cn(tu0, dx, dy, dt_cn, pasos_cn)
sol_adi = resolver_adi(tu0, dx, dy, dt_adi, pasos_adi)
err_ftcs = error_l2(sol_ftcs[-1], u_exact)
err_cn = error_l2(sol_cn[-1], u_exact)
err_adi = error_l2(sol_adi[-1], u_exact)

print("\n=== Comparación métodos con solución analítica ===")
print(f"Error L2 FTCS: {err_ftcs:.3e}")
print(f"Error L2 Crank-Nicolson: {err_cn:.3e}")
print(f"Error L2 ADI: {err_adi:.3e}")

plt.figure(figsize=(15,8))
plt.subplot(2,2,1); plt.title("FTCS numérico"); plt.imshow(sol_ftcs[-1], origin='lower', cmap='viridis'); plt.colorbar()
plt.subplot(2,2,2); plt.title("Crank-Nicolson numérico"); plt.imshow(sol_cn[-1], origin='lower', cmap='viridis'); plt.colorbar()
plt.subplot(2,2,3); plt.title("ADI numérico"); plt.imshow(sol_adi[-1], origin='lower', cmap='viridis'); plt.colorbar()
plt.subplot(2,2,4); plt.title("Solución analítica"); plt.imshow(u_exact, origin='lower', cmap='viridis'); plt.colorbar()
plt.tight_layout()
plt.show()

#------------------ Ejemplo real: placa caliente arriba, fría abajo ---------------#
print("\n=== Ejemplo realista: placa con borde superior caliente ===")
nx, ny = 60, 60
x2, y2, dx2, dy2 = inicializar_dominio(nx, ny, 1.0, 1.0)
u0_real = np.zeros((nx, ny))  # Todo frío
u0_real[-1, :] = 1.0         # Borde superior caliente

# FTCS para el ejemplo real
sol_ftcs_real = resolver_ftcs(u0_real, dx2, dy2, 0.0001, 100, tipo_frontera='dirichlet', valor_frontera=0.0)
for u in sol_ftcs_real:
    u[-1, :] = 1.0

# CN para el ejemplo real
sol_cn_real = resolver_cn(u0_real, dx2, dy2, 0.001, 20, tipo_frontera='dirichlet', valor_frontera=0.0)
for u in sol_cn_real:
    u[-1, :] = 1.0

# ADI para el ejemplo real
sol_adi_real = resolver_adi(u0_real, dx2, dy2, 0.002, 10, tipo_frontera='dirichlet', valor_frontera=0.0)
for u in sol_adi_real:
    u[-1, :] = 1.0

plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.title("FTCS - borde superior caliente")
plt.imshow(sol_ftcs_real[-1], origin='lower', cmap='plasma'); plt.colorbar()
plt.subplot(1,3,2)
plt.title("Crank-Nicolson - borde superior caliente")
plt.imshow(sol_cn_real[-1], origin='lower', cmap='plasma'); plt.colorbar()
plt.subplot(1,3,3)
plt.title("ADI - borde superior caliente")
plt.imshow(sol_adi_real[-1], origin='lower', cmap='plasma'); plt.colorbar()
plt.tight_layout()
plt.show()

print("\nObserva los gradientes de temperatura generados al modelar una placa mantenida caliente en el borde superior y fría en el resto. Resultados congruentes entre los tres métodos.")
