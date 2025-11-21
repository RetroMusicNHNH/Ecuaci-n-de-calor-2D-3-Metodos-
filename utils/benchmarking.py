"""
Benchmarking y análisis de escalabilidad para FTCS, Crank-Nicolson y ADI (corregido y robusto)
"""
import sys
sys.path.append('./')

import numpy as np
import matplotlib.pyplot as plt
import time
from src.solucionadores import resolver_ftcs, resolver_cn, resolver_adi


def condicion_inicial_seno(X, Y):
    return np.sin(np.pi * X) * np.sin(np.pi * Y)

def solucion_exacta(X, Y, t, alpha=1.0):
    decay = np.exp(-2 * np.pi**2 * alpha * t)
    return decay * np.sin(np.pi * X) * np.sin(np.pi * Y)

def calcular_error_l2(u_num, u_exact, dx, dy):
    error = u_num[1:-1, 1:-1] - u_exact[1:-1, 1:-1]
    return np.sqrt(np.sum(error**2) * dx * dy)

def ajustar_pasos_dt(T_final, dt):
    """Ajusta el número de pasos y dt para cubrir T_final exacto"""
    pasos = int(np.ceil(T_final / dt))
    dt_ajustado = T_final / pasos
    return pasos, dt_ajustado

def benchmarking_tiempo(tamaños_malla, alpha=1.0, T_final=0.01):
    resultados = {
        'FTCS': {'tiempos': [], 'errores': [], 'tamaños': [], 'pasos': []},
        'Crank-Nicolson': {'tiempos': [], 'errores': [], 'tamaños': [], 'pasos': []},
        'ADI': {'tiempos': [], 'errores': [], 'tamaños': [], 'pasos': []}
    }
    print("Ejecutando benchmarking de tiempo y precisión...")
    print("-" * 60)
    for N in tamaños_malla:
        Lx, Ly = 1.0, 1.0
        dx = Lx / (N - 1)
        dy = Ly / (N - 1)
        # FTCS requiere dt pequeño (CFL)
        dt_ftcs = min(0.9 * dx**2 / (4 * alpha), T_final)
        pasos_ftcs, dt_ftcs = ajustar_pasos_dt(T_final, dt_ftcs)
        # CN y ADI usan mayor dt, ajustamos igual
        dt_implicito = min(4 * dt_ftcs, T_final)
        pasos_implicito, dt_implicito = ajustar_pasos_dt(T_final, dt_implicito)
        x = np.linspace(0, Lx, N)
        y = np.linspace(0, Ly, N)
        X, Y = np.meshgrid(x, y)
        u0 = condicion_inicial_seno(X, Y)
        u_exact = solucion_exacta(X, Y, T_final, alpha)
        print(f"\nMalla: {N}x{N} (Total: {N*N} puntos)")
        # FTCS
        try:
            if dt_ftcs > dx**2 / (4 * alpha):
                print(f"  Advertencia: FTCS inestable. CFL > 0.25")
            t_inicio = time.time()
            sol_ftcs = resolver_ftcs(u0, dx, dy, dt_ftcs, pasos_ftcs, alpha)
            t_ftcs = time.time() - t_inicio
            error_ftcs = calcular_error_l2(sol_ftcs[-1], u_exact, dx, dy)
            resultados['FTCS']['tiempos'].append(t_ftcs)
            resultados['FTCS']['errores'].append(error_ftcs)
            resultados['FTCS']['tamaños'].append(N)
            resultados['FTCS']['pasos'].append(pasos_ftcs)
            print(f"  FTCS: {t_ftcs:.4f} s | Error L2: {error_ftcs:.2e} | Pasos: {pasos_ftcs}")
        except Exception as e:
            print(f"  FTCS: Error - {e}")
        # CN
        try:
            t_inicio = time.time()
            sol_cn = resolver_cn(u0, dx, dy, dt_implicito, pasos_implicito, alpha)
            t_cn = time.time() - t_inicio
            error_cn = calcular_error_l2(sol_cn[-1], u_exact, dx, dy)
            resultados['Crank-Nicolson']['tiempos'].append(t_cn)
            resultados['Crank-Nicolson']['errores'].append(error_cn)
            resultados['Crank-Nicolson']['tamaños'].append(N)
            resultados['Crank-Nicolson']['pasos'].append(pasos_implicito)
            print(f"  Crank-Nicolson: {t_cn:.4f} s | Error L2: {error_cn:.2e} | Pasos: {pasos_implicito}")
        except Exception as e:
            print(f"  Crank-Nicolson: Error - {e}")
        # ADI
        try:
            t_inicio = time.time()
            sol_adi = resolver_adi(u0, dx, dy, dt_implicito, pasos_implicito, alpha)
            t_adi = time.time() - t_inicio
            error_adi = calcular_error_l2(sol_adi[-1], u_exact, dx, dy)
            resultados['ADI']['tiempos'].append(t_adi)
            resultados['ADI']['errores'].append(error_adi)
            resultados['ADI']['tamaños'].append(N)
            resultados['ADI']['pasos'].append(pasos_implicito)
            print(f"  ADI: {t_adi:.4f} s | Error L2: {error_adi:.2e} | Pasos: {pasos_implicito}")
        except Exception as e:
            print(f"  ADI: Error - {e}")
    print("\n" + "=" * 60)
    print("Benchmarking completado")
    return resultados

def analisis_escalabilidad(tamaños_malla, alpha=1.0, pasos_fijos=100):
    escalabilidad = {
        'FTCS': {'tiempos': [], 'tamaños': [], 'puntos_totales': []},
        'Crank-Nicolson': {'tiempos': [], 'tamaños': [], 'puntos_totales': []},
        'ADI': {'tiempos': [], 'tamaños': [], 'puntos_totales': []}
    }
    print("\nEjecutando análisis de escalabilidad...")
    print("-" * 60)
    for N in tamaños_malla:
        Lx, Ly = 1.0, 1.0
        dx = Lx / (N - 1)
        dy = Ly / (N - 1)
        dt = min(0.9 * dx ** 2 / (4 * alpha), 0.01)
        x = np.linspace(0, Lx, N)
        y = np.linspace(0, Ly, N)
        X, Y = np.meshgrid(x, y)
        u0 = condicion_inicial_seno(X, Y)
        puntos_totales = N * N
        print(f"\nMalla: {N}x{N} ({puntos_totales} puntos)")
        # FTCS
        try:
            t_inicio = time.time()
            resolver_ftcs(u0, dx, dy, dt, pasos_fijos, alpha)
            t_ftcs = time.time() - t_inicio
            escalabilidad['FTCS']['tiempos'].append(t_ftcs)
            escalabilidad['FTCS']['tamaños'].append(N)
            escalabilidad['FTCS']['puntos_totales'].append(puntos_totales)
            print(f"  FTCS: {t_ftcs:.4f} s")
        except Exception as e:
            print(f"  FTCS: Error - {e}")
        # Crank-Nicolson
        try:
            t_inicio = time.time()
            resolver_cn(u0, dx, dy, dt, pasos_fijos, alpha)
            t_cn = time.time() - t_inicio
            escalabilidad['Crank-Nicolson']['tiempos'].append(t_cn)
            escalabilidad['Crank-Nicolson']['tamaños'].append(N)
            escalabilidad['Crank-Nicolson']['puntos_totales'].append(puntos_totales)
            print(f"  Crank-Nicolson: {t_cn:.4f} s")
        except Exception as e:
            print(f"  Crank-Nicolson: Error - {e}")
        # ADI
        try:
            t_inicio = time.time()
            resolver_adi(u0, dx, dy, dt, pasos_fijos, alpha)
            t_adi = time.time() - t_inicio
            escalabilidad['ADI']['tiempos'].append(t_adi)
            escalabilidad['ADI']['tamaños'].append(N)
            escalabilidad['ADI']['puntos_totales'].append(puntos_totales)
            print(f"  ADI: {t_adi:.4f} s")
        except Exception as e:
            print(f"  ADI: Error - {e}")
    print("\n" + "=" * 60)
    print("Análisis de escalabilidad completado")
    return escalabilidad

def graficar_resultados(resultados, escalabilidad, guardar=True):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    # 1
    ax = axes[0, 0]
    for metodo in ['FTCS', 'Crank-Nicolson', 'ADI']:
        if resultados[metodo]['tiempos']:
            ax.plot(resultados[metodo]['tamaños'], 
                   resultados[metodo]['tiempos'], 
                   marker='o', label=metodo, linewidth=2)
    ax.set_xlabel('Tamaño de malla N', fontsize=11)
    ax.set_ylabel('Tiempo de ejecución (s)', fontsize=11)
    ax.set_title('Benchmarking: Tiempo vs Tamaño de Malla', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    # 2
    ax = axes[0, 1]
    for metodo in ['FTCS', 'Crank-Nicolson', 'ADI']:
        if resultados[metodo]['errores']:
            ax.semilogy(resultados[metodo]['tamaños'], 
                       resultados[metodo]['errores'], 
                       marker='s', label=metodo, linewidth=2)
    ax.set_xlabel('Tamaño de malla N', fontsize=11)
    ax.set_ylabel('Error L2', fontsize=11)
    ax.set_title('Precisión: Error L2 vs Tamaño de Malla', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    # 3
    ax = axes[1, 0]
    for metodo in ['FTCS', 'Crank-Nicolson', 'ADI']:
        if escalabilidad[metodo]['tiempos']:
            ax.loglog(escalabilidad[metodo]['puntos_totales'], 
                     escalabilidad[metodo]['tiempos'], 
                     marker='^', label=metodo, linewidth=2)
    ax.set_xlabel('Puntos totales (N²)', fontsize=11)
    ax.set_ylabel('Tiempo de ejecución (s)', fontsize=11)
    ax.set_title('Escalabilidad: Tiempo vs Puntos Totales', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    # 4
    ax = axes[1, 1]
    if resultados['FTCS']['tiempos'] and len(resultados['FTCS']['tiempos']) > 0:
        tamaños = resultados['FTCS']['tamaños']
        for metodo in ['Crank-Nicolson', 'ADI']:
            if resultados[metodo]['tiempos']:
                eficiencia = []
                for i in range(len(resultados['FTCS']['tiempos'])):
                    t_ftcs = resultados['FTCS']['tiempos'][i]
                    t_comp = resultados[metodo]['tiempos'][i] if i < len(resultados[metodo]['tiempos']) else None
                    if t_comp is not None and t_comp > 0:
                        eficiencia.append(t_ftcs / t_comp)
                    else:
                        eficiencia.append(0)
                ax.plot(tamaños, eficiencia, marker='d', label=f'{metodo} vs FTCS', linewidth=2)
        ax.axhline(y=1.0, color='r', linestyle='--', label='Referencia (1.0)')
        ax.set_xlabel('Tamaño de malla N', fontsize=11)
        ax.set_ylabel('Eficiencia relativa', fontsize=11)
        ax.set_title('Eficiencia Relativa (FTCS como base)', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if guardar:
        plt.savefig('benchmarking_resultados.png', dpi=300, bbox_inches='tight')
        print("\nGráficos guardados: benchmarking_resultados.png")
    plt.show()

def ejecutar_benchmarking_completo():
    tamaños_benchmarking = [10, 20, 30, 40, 50]
    tamaños_escalabilidad = [10, 20, 40, 60, 80, 100]
    print("=" * 60)
    print("BENCHMARKING Y ANALISIS DE ESCALABILIDAD")
    print("Ecuación de Calor 2D: FTCS, Crank-Nicolson, ADI")
    print("=" * 60)
    resultados = benchmarking_tiempo(tamaños_benchmarking)
    escalabilidad = analisis_escalabilidad(tamaños_escalabilidad)
    graficar_resultados(resultados, escalabilidad)
    print("\n" + "=" * 60)
    print("RESUMEN DE RESULTADOS")
    print("=" * 60)
    print("| Método          | Error Promedio | Tiempo Promedio (s) |")
    print("|-----------------|----------------|---------------------|")
    for metodo in ['FTCS', 'Crank-Nicolson', 'ADI']:
        if resultados[metodo]['tiempos']:
            err_avg = np.mean(resultados[metodo]['errores'])
            t_avg = np.mean(resultados[metodo]['tiempos'])
            print(f"| {metodo:15} | {err_avg:1.2e}     | {t_avg:10.4f}         |")
    print("\n" + "=" * 60)

if __name__ == "__main__":
    ejecutar_benchmarking_completo()
