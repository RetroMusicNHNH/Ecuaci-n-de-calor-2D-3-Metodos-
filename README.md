# Ecuación de Calor 2D - Tres Métodos Numéricos

## Descripción

Implementación y análisis comparativo de métodos de diferencias finitas para resolver la ecuación de calor bidimensional.

**Métodos implementados:**
- FTCS (Forward Time Central Space) - Explícito
- Crank-Nicolson - Implícito
- ADI (Alternating Direction Implicit)

## Autor

Luis Enrique Reyes García (C36518)  
Curso: MO-0014 Álgebra Lineal Numérica  
Profesor: Jorge Luis Salazar Chaves  
Universidad de Costa Rica - II Ciclo 2025

## Estructura del Proyecto

```
Ecuacion-de-calor-2D-3-Metodos/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── solucionadores.py    # FTCS, CN, ADI
│   ├── condiciones.py       # Frontera, iniciales
│   └── validacion.py        # Tests y errores
├── ejemplos/
│   ├── ejemplo_ftcs.py
│   ├── ejemplo_cn.py
│   └── ejemplo_adi.py
├── tests/
│   └── test_convergencia.py
└── notebooks/
    └── demo_completa.ipynb
```

## Instalación

```bash
pip install -r requirements.txt
```

## Uso Rápido

```python
from src.solucionadores import resolver_ftcs
from src.condiciones import inicializar_dominio, temperatura_inicial

# Configurar dominio
x, y, dx, dy = inicializar_dominio(nx=50, ny=50)
u0 = temperatura_inicial(x, y, tipo='gaussiana')

# Resolver
resultado = resolver_ftcs(u0, dx, dy, dt=0.001, pasos=100)
```

## Fase 1 - Desarrollo de Métodos

- [x] Estructura del proyecto
- [ ] Módulo de condiciones
- [ ] Módulo de solucionadores
- [ ] Módulo de validación
- [ ] Ejemplos de uso
- [ ] Tests de convergencia

## Referencias

Ver anteproyecto adjunto para referencias completas.
