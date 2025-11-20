"""Benchmarking y análisis de escalabilidad para FTCS, Crank-Nicolson y ADI"""

import sys
sys.path.append('./')

import numpy as np
import matplotlib.pyplot as plt
import time
from src.solucionadores import resolver_ftcs, resolver_cn, resolver_adi
from src.condiciones import condicion_inicial_seno


def solucion_exacta(X, Y, t, alpha=1.0):
    """Solución exacta para validación"""
    decay = np.exp(-2 * np.pi**2 * alpha * t)
    return decay * np.sin(np.pi * X) * np.sin(np.pi * Y)


def calcular_error_l2(u_num, u_exact, dx, dy):
    """Calcula error L2"""
    error = u_num[1:-1, 1:-1] - u_exact[1:-1, 1:-1]
    return np.sqrt(np.sum(error**2) * dx * dy)


def benchmarking_tiempo(tamaños_malla, alpha=1.0, T_final=0.01):
    """
    Benchmarking de tiempo de ejecución para los tres métodos.
    
    Args:
        tamaños_malla: lista de tamaños N (malla NxN)
        alpha: difusividad térmica
        T_final: tiempo final de simulación
    
    Returns:
        dict con tiempos y errores por método
    """
    resultados = {
        'FTCS': {'tiempos': [], 'errores': [], 'tamaños': []},
        'Crank-Nicolson': {'tiempos': [], 'errores': [], 'tamaños': []},
        'ADI': {'tiempos': [], 'errores': [], 'tamaños': []}
    }
    
    print("Ejecutando benchmarking de tiempo y precision...")
    print("-" * 60)
    
    for N in tamaños_malla:
        Lx, Ly = 1.0, 1.0
        dx = Lx / (N - 1)
        dy = Ly / (N - 1)
        
        # FTCS requiere dt pequeño por estabilidad
        dt_ftcs = 0.9 * dx**2 / (4 * alpha)
        pasos_ftcs = int(T_final / dt_ftcs)
        
        # CN y ADI pueden usar dt mayor
        dt_implicito = 4 * dt_ftcs
        pasos_implicito = int(T_final / dt_implicito)
        
        # Condición inicial
        x = np.linspace(0, Lx, N)
        y = np.linspace(0, Ly, N)
        X, Y = np.meshgrid(x, y)
        u0 = condicion_inicial_seno(X, Y)
        
        # Solución exacta en T_final
        u_exact = solucion_exacta(X, Y, T_final, alpha)
        
        print(f"\nMalla: {N}x{N} (Total: {N*N} puntos)")
        
        # FTCS
        try:
            t_inicio = time.time()
            sol_ftcs = resolver_ftcs(u0, dx, dy, dt_ftcs, pasos_ftcs, alpha)
            t_ftcs = time.time() - t_inicio
            error_ftcs = calcular_error_l2(sol_ftcs[-1], u_exact, dx, dy)
            
            resultados['FTCS']['tiempos'].append(t_ftcs)
            resultados['FTCS']['errores'].append(error_ftcs)
            resultados['FTCS']['tamaños'].append(N)
            
            print(f"  FTCS: {t_ftcs:.4f} s | Error L2: {error_ftcs:.2e} | Pasos: {pasos_ftcs}")
        except Exception as e:
            print(f"  FTCS: Error - {e}")
        
        # Crank-Nicolson
        try:
            t_inicio = time.time()
            sol_cn = resolver_cn(u0, dx, dy, dt_implicito, pasos_implicito, alpha)
            t_cn = time.time() - t_inicio
            error_cn = calcular_error_l2(sol_cn[-1], u_exact, dx, dy)
            
            resultados['Crank-Nicolson']['tiempos'].append(t_cn)
            resultados['Crank-Nicolson']['errores'].append(error_cn)
            resultados['Crank-Nicolson']['tamaños'].append(N)
            
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
            
            print(f"  ADI: {t_adi:.4f} s | Error L2: {error_adi:.2e} | Pasos: {pasos_implicito}")
        except Exception as e:
            print(f"  ADI: Error - {e}")
    
    print("\n" + "=" * 60)
    print("Benchmarking completado")
    return resultados


def analisis_escalabilidad(tamaños_malla, alpha=1.0, pasos_fijos=100):
    """
    Análisis de escalabilidad: tiempo vs tamaño de malla.
    
    Args:
        tamaños_malla: lista de tamaños N
        alpha: difusividad
        pasos_fijos: número fijo de pasos temporales
    
    Returns:
        dict con datos de escalabilidad
    """
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
        dt = 0.9 * dx**2 / (4 * alpha)
        
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
    """
    Genera gráficos de benchmarking y escalabilidad.
    
    Args:
        resultados: datos de benchmarking
        escalabilidad: datos de escalabilidad
        guardar: si True, guarda las figuras
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Gráfico 1: Tiempo vs Tamaño de malla
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
    
    # Gráfico 2: Error vs Tamaño de malla
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
    
    # Gráfico 3: Escalabilidad (Tiempo vs Puntos totales)
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
    
    # Gráfico 4: Eficiencia relativa
    ax = axes[1, 1]
    if resultados['FTCS']['tiempos'] and len(resultados['FTCS']['tiempos']) > 0:
        tamaños = resultados['FTCS']['tamaños']
        
        for metodo in ['Crank-Nicolson', 'ADI']:
            if resultados[metodo]['tiempos']:
                eficiencia = [
                    resultados['FTCS']['tiempos'][i] / resultados[metodo]['tiempos'][i]
                    if i < len(resultados[metodo]['tiempos']) else 0
                    for i in range(len(resultados['FTCS']['tiempos']))
                ]
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
    """Ejecuta benchmarking completo y genera reportes"""
    
    # Configuración
    tamaños_benchmarking = [10, 20, 30, 40, 50]
    tamaños_escalabilidad = [10, 20, 40, 60, 80, 100]
    
    print("=" * 60)
    print("BENCHMARKING Y ANALISIS DE ESCALABILIDAD")
    print("Ecuación de Calor 2D: FTCS, Crank-Nicolson, ADI")
    print("=" * 60)
    
    # Benchmarking de tiempo y precisión
    resultados = benchmarking_tiempo(tamaños_benchmarking)
    
    # Análisis de escalabilidad
    escalabilidad = analisis_escalabilidad(tamaños_escalabilidad)
    
    # Generar gráficos
    graficar_resultados(resultados, escalabilidad)
    
    # Resumen en consola
    print("\n" + "=" * 60)
    print("RESUMEN DE RESULTADOS")
    print("=" * 60)
    
    for metodo in ['FTCS', 'Crank-Nicolson', 'ADI']:
        if resultados[metodo]['tiempos']:
            print(f"\n{metodo}:")
            print(f"  Tiempo promedio: {np.mean(resultados[metodo]['tiempos']):.4f} s")
            print(f"  Error promedio L2: {np.mean(resultados[metodo]['errores']):.2e}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    ejecutar_benchmarking_completo()
