# Análisis Comparativo de Operadores de Cruza en Algoritmos Genéticos para Optimización Continua

**Autor:** Martínez Martínez Geovani — 320141384

**Materia:** Cómputo Evolutivo

**Universidad:** UNAM — Facultad de Ciencias

**Fecha:** Noviembre 2025

---

## Descripción

Este proyecto implementa un **Algoritmo Genético (AG)** con codificación real para comparar **4 operadores de cruza** en problemas de optimización continua. El objetivo es analizar cómo cada operador afecta la  **convergencia** ,  **calidad final** , **diversidad poblacional** y  **costo computacional** .

### **Operadores de Cruza Comparados:**

| Operador           | Característica                         | Parámetro  |
| ------------------ | --------------------------------------- | ----------- |
| **Un Punto** | Corte simple en posición aleatoria     | —          |
| **Uniforme** | Cada gen heredado independientemente    | —          |
| **BLX-α**   | Blend: exploración alrededor de padres | α = 0.5    |
| **SBX**      | Simulated Binary Crossover              | η_c = 10.0 |

---

## 📁 Estructura del Repositorio

```
AG-OC/
├─ src/
│  ├─ componentes/
│  │  ├─ main_ga.py                      # Script principal (experimentos)
│  │  ├─ funciones.py                    # Benchmarks (Sphere, Rastrigin, Rosenbrock)
│  │  │
│  │  ├─ cruza_un_punto.py               # Operador: Un Punto
│  │  ├─ cruza_uniforme.py               # Operador: Uniforme
│  │  ├─ cruza_blx.py                    # Operador: BLX-α
│  │  ├─ cruza_sbx.py                    # Operador: SBX
│  │  │
│  │  ├─ seleccion_ruleta.py             # Selección por ruleta + transformación aptitud
│  │  ├─ mutacion_real.py                # Mutación uniforme en reales
│  │  ├─ reemplazo_peores.py             # Reemplazo generacional + elitismo
│  │  ├─ calcular_diversidad.py          # Métrica de diversidad poblacional
│  │  │
│  │  ├─ graficas_convergencia.py        # Visualización: Convergencia por generación
│  │  ├─ graficas_boxplot.py             # Visualización: Distribución final (boxplots)
│  │  ├─ graficas_diversidad.py          # Visualización: Pérdida de diversidad
│  │  └─ graficas_tiempo.py              # Visualización: Costo computacional
│  │
│  └─ README.md (este archivo)
│
└─ output/
   ├─ resultados/
   │  ├─ resultados_ga_sphere_rastrigin_rosenbrock.csv          # Resumen: una fila por ejecución
   │  └─ resultados_ga_sphere_rastrigin_rosenbrock_curvas.csv   # Curvas: una fila por generación
   │
   └─ graficas/
      ├─ convergencia_sphere.png          # Convergencia - Función Sphere
      ├─ convergencia_rastrigin.png       # Convergencia - Función Rastrigin
      ├─ convergencia_rosenbrock.png      # Convergencia - Función Rosenbrock
      │
      ├─ boxplot_calidad_sphere.png       # Distribución final - Sphere
      ├─ boxplot_calidad_rastrigin.png    # Distribución final - Rastrigin
      ├─ boxplot_calidad_rosenbrock.png   # Distribución final - Rosenbrock
      │
      ├─ diversidad_sphere.png            # Diversidad - Sphere
      ├─ diversidad_rastrigin.png         # Diversidad - Rastrigin
      ├─ diversidad_rosenbrock.png        # Diversidad - Rosenbrock
      │
      ├─ tiempo_sphere.png                # Costo computacional - Sphere
      ├─ tiempo_rastrigin.png             # Costo computacional - Rastrigin
      └─ tiempo_rosenbrock.png            # Costo computacional - Rosenbrock
```

---

## Inicio

### **Requisitos**

```bash
Python >= 3.8
pip install numpy pandas matplotlib seaborn scikit-learn scipy
```

### **1. Ejecutar los Experimentos**

```bash
cd src/componentes
python main_ga.py
```

**Opciones (incluidas en main_ga.py):**

```bash
# Modo: Semillas independientes (default)
python main_ga.py

# Modo: Semillas por bloques (para reproducibilidad exacta)
python main_ga.py -s 42
python main_ga.py --seed 42
```

**Parámetros (configurables en `main_ga.py` línea final):**

* `funciones`: Lista de funciones a optimizar
* `cruzas`: Operadores a comparar
* `dim`: Dimensión del problema (default: 10)
* `tam_pob`: Tamaño de población (default: 100)
* `generaciones`: Máximo de generaciones (default: 1000)
* `repeticiones`: Corridas por configuración (default: 30)

### **2. Generar Gráficas**

```bash
# Convergencia
python graficas_convergencia.py

# Boxplots (Distribución final)
python graficas_boxplot.py

# Diversidad poblacional
python graficas_diversidad.py

# Costo computacional
python graficas_tiempo.py
```

---

## Configuración Experimental

### **Parámetros Fijos (AG):**

| Parámetro                    | Valor               | Justificación                    |
| ----------------------------- | ------------------- | --------------------------------- |
| **Dimensión**          | 10                  | Estándar, manejable              |
| **Población**          | 100                 | Balance exploración/explotación |
| **Generaciones**        | 1,000               | Convergencia suficiente           |
| **Prob. Cruza**         | 0.9                 | Estándar en literatura           |
| **Prob. Mutación/Gen** | 1/dim ≈ 0.1        | Heurística común                |
| **Amplitud Mutación**  | 0.1 × rango        | 10% del dominio                   |
| **Elitismo**            | 1                   | Preserva al mejor                 |
| **Reemplazo**           | 100% (Generacional) | Presión selectiva moderada       |

### **Funciones de Prueba:**

| Función             | Dominio         | Óptimo      | Características                  |
| -------------------- | --------------- | ------------ | --------------------------------- |
| **Sphere**     | [-5.12, 5.12]   | 0.0          | Unimodal, convexa, suave          |
| **Rastrigin**  | [-5.12, 5.12]   | 0.0          | Altamente multimodal, oscilatoria |
| **Rosenbrock** | [-2.048, 2.048] | 0.0 (en x=1) | Valle estrecho, asimétrica       |

### **Repeticiones y Reproducibilidad:**

* **30 repeticiones** por (función, operador) pair
* **Semillas fijas** : 1000 × índice + 123
* **CSV generados** : Contienen TODAS las métricas para el análisis

---

## Resultados Esperados

### **Archivos Generados**

**Archivo de Resumen** (`resultados_ga_*.csv`):

* Una fila por ejecución
* Columnas: función, operador, métricas finales, tiempo

**Archivo de Curvas** (`resultados_ga_*_curvas.csv`):

* Una fila por generación
* Columnas: función, operador, generación, mejor, promedio, diversidad

### **Gráficas Generadas**

| Gráfica               | Pregunta                        | Insight                           |
| ---------------------- | ------------------------------- | --------------------------------- |
| **Convergencia** | ¿Quién converge más rápido? | Velocidad de búsqueda            |
| **Boxplot**      | ¿Quién es más confiable?     | Robustez y calidad final          |
| **Diversidad**   | ¿Quién mantiene exploración? | Balance exploración/explotación |
| **Tiempo**       | ¿Quién es más eficiente?     | Costo computacional               |

---

## Métricas Utilizadas

### **1. Velocidad de Convergencia**

```
Métrica: mejor_generacion (mejor valor encontrado hasta gen G)
Visualización: Curva log de convergencia
```

### **2. Calidad Final**

```
Métricas:
  - mejor_final: Mejor solución alcanzada
  - promedio_final: Promedio poblacional final
  - peor_final: Peor solución final
Visualización: Boxplot con escala logarítmica
```

### **3. Diversidad Poblacional**

```
Métrica: Desv. Estándar promediada por dimensión
Fórmula: avg(std(población[:, i])) para i en dimensiones
Interpretación:
  - Alta: Población esparcida, buena exploración
  - Baja: Población convergida, posible estancamiento
Visualización: Curva log por generación
```

### **4. Robustez**

```
Base: 30 repeticiones con semillas diferentes
Análisis: Mediana, cuartiles, desviación estándar
```

### **5. Eficiencia Computacional**

```
Métrica: Tiempo total de ejecución (segundos)
Incluye: Evaluaciones, selección, cruza, mutación, reemplazo
Visualización: Barras agrupadas por función
```

---

## Referencias

* Deb, K., & Agrawal, R. (1995).  *Simulated binary crossover for continuous search space* . Complex Systems, 9(3), 1-15.
* Eiben, A. E., & Smith, J. E. (2003).  *Introduction to evolutionary computing* . Springer.
* Goldberg, D. E. (1989).  *Genetic algorithms in search, optimization, and machine learning* . Addison-Wesley.
