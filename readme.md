# Comparación operadores cruza AG - optimización continua

**Martínez Martínez Geovani — 320141384**  

---

## Descripción
En esta tarea implementamos **Evolución Diferencial (DE)** para optimización continua en **dimensión 10**.  
Se exploran **3 variantes DE** (rand/1/bin, best/1/bin, rand/2/bin) con diferentes valores de **F** y **CR**, y se prueban en **5 funciones** de optimización.

---

## Estructura del repositorio
```
AG-OC/
├─ output/
│  └─ experimentos_de/
│     ├─ DE_rand_1_bin_F0.5_CR0.1/
│     ├─ DE_rand_1_bin_F0.5_CR0.5/
│     ├─ DE_rand_1_bin_F0.5_CR0.9/
│     ├─ ... (27 configuraciones)
│     └─ DE_rand_2_bin_F1.0_CR0.9/
|  └─ graficas/
│     ├─ boxplot_ackley.png
|     ├─ boxplot_ackley.png
|     ├─ ...
|     ├─ estrategias_ackley.png
|     ├─ ...
|     ├─ estrategias_sphere.png
|     ├─ sensibilidad_ackley.png
|     ├─ ...
|     ├─ sensibilidad_sphere.png
└─ src/
   └─ componentes/
      ├─ MainDE.py                    # Script principal de experimentación
      ├─ Individuo.py                 # Representación de vectores
      ├─ EvolucionDiferencial.py      # Las 3 variantes DE
      ├─ ManejoLimites.py             # Manejo de restricciones
      ├─ Funciones.py                 # 5 funciones de prueba
      │
      └─ graficacion/
         ├─ graficar_resultados.py
         └─ graficar_boxplots.py
```

---

## Requisitos
- **Python 3.x**
- Paquetes:
  ```bash
  pip install numpy pandas matplotlib seaborn
  ```

---

## Uso

### 1) Ejecutar experimentos (genera los CSV)

Ejecuta `MainDE.py` **desde `src/componentes/`**.

#### **Opciones de ejecución:**

**a) Experimentación completa (5 funciones, 1,350 ejecuciones):**
```bash
cd src/componentes
python MainDE.py -c --ejecuciones 10
```

**b) Funciones específicas:**
```bash
# Terminal 1
python MainDE.py sphere ackley griewank --ejecuciones 10

# Terminal 2
python MainDE.py rastrigin rosenbrock --ejecuciones 10
```

**c) Con semilla fija (reproducibilidad):**
```bash
# Una persona:
python MainDE.py -c --ejecuciones 10 --semilla 42

# Dividido:
python MainDE.py sphere ackley griewank --ejecuciones 10 --semilla 42
python MainDE.py rastrigin rosenbrock --ejecuciones 10 --semilla 42
```

**Parámetros:**
- `--ejecuciones N`: Número de repeticiones por configuración (default: 10, mínimo: 10)
- `--semilla N`: Semilla base para reproducibilidad (default: generada automáticamente)
- `-c`: Flag para experimentación completa (todas las funciones)
- Funciones válidas: `sphere`, `ackley`, `griewank`, `rastrigin`, `rosenbrock`

---

### 2) Qué se genera

Por cada configuración (variante + F + CR) y función, se generan:

**a) Curvas de evolución individuales:**
```
curva_<funcion>_<variante>_F<valor>_CR<valor>_sem<semilla>.csv
```
Columnas: `generacion, mejor, promedio, peor`

**b) Resúmenes individuales:**
```
resumen_<funcion>_<variante>_F<valor>_CR<valor>_sem<semilla>.csv
```
Columnas: `funcion, variante, F, CR, poblacion, dimension, max_evals, semilla, mejor, promedio, mediana, peor, std`

**c) Resúmenes consolidados (múltiples repeticiones):**
```
multi_<funcion>_<variante>_F<valor>_CR<valor>.csv
```
Columnas: `semilla, mejor, promedio, mediana, peor, std`

**Ubicación:** `output/experimentos_de/<configuracion>/`

---

### 3) Analizar resultados (gráficas y tablas)

```bash
# Ejecutar desde la raíz del repositorio (Syrion6/)
python src/graficacion/graficar_resultados.py
python src/graficacion/graficar_boxplots.py
```

---

## Configuración experimental

### **Parámetros fijos:**
- Dimensión: **D = 10**
- Población: **Npob = 100**
- Evaluaciones máximas: **300,000**
- Generaciones: **3,000** (300,000 / 100)
- Método de límites: **Saturación**

### **Parámetros variables:**
- **Variantes:** 3 (DE/rand/1/bin, DE/best/1/bin, DE/rand/2/bin)
- **Factor F:** 3 valores (0.5, 0.75, 1.0)
- **Tasa CR:** 3 valores (0.1, 0.5, 0.9)
- **Total configuraciones:** 3 × 3 × 3 = **27**

### **Funciones de prueba:**

| Función | Límites | Óptimo | Características |
|---------|---------|--------|-----------------|
| Sphere | [-100, 100] | 0.0 | Unimodal, convexa |
| Ackley | [-32, 32] | 0.0 | Multimodal |
| Griewank | [-600, 600] | 0.0 | Multimodal |
| Rastrigin | [-5.12, 5.12] | 0.0 | Altamente multimodal |
| Rosenbrock | [-5, 10] | 0.0 | Valle estrecho |

---

## Implementación

### **Variantes DE implementadas:**

#### **1. DE/rand/1/bin (Clásica - Ejercicio 1.a)**
```
v_i = x_r1 + F * (x_r2 - x_r3)
```
- Balance entre exploración y explotación
- Vector base aleatorio

#### **2. DE/best/1/bin (Geovani - Ejercicio 1.c)**
```
v_i = x_best + F * (x_r2 - x_r3)
```
- Variante explotadora
- Convergencia más rápida
- Usa el mejor individuo como base

#### **3. DE/rand/2/bin (Osdan - Ejercicio 1.c)**
```
v_i = x_r1 + F * (x_r2 - x_r3) + F * (x_r4 - x_r5)
```
- Variante exploradora
- Mayor diversidad
- Requiere mínimo 6 individuos

### **Manejo de límites (Ejercicio 1.b):**
- **Saturación (Clamping):** Método principal utilizado
- Rebote (Reflection)
- Reinicio aleatorio

---

## Notas importantes

- **Semillas:** Si no se especifica `--semilla`, se genera automáticamente con `secrets.randbits(32)`
- **Reproducibilidad:** Usar la misma semilla base garantiza resultados idénticos
- **División de trabajo:** Ambas personas deben usar la misma semilla para resultados consistentes
- **Tiempo estimado:** ~6-7 segundos por ejecución en hardware moderno
- **Archivos generados:** Cada configuración genera archivos por función (no hay conflictos al dividir)

---

## Estructura de resultados

```
output/experimentos_de/
├─ DE_rand_1_bin_F0.5_CR0.1/
│  ├─ curva_sphere_DE_rand_1_bin_F0.5_CR0.1_sem42.csv
│  ├─ curva_sphere_DE_rand_1_bin_F0.5_CR0.1_sem43.csv
│  ├─ ...
│  ├─ resumen_sphere_DE_rand_1_bin_F0.5_CR0.1_sem42.csv
│  ├─ ...
│  └─ multi_sphere_DE_rand_1_bin_F0.5_CR0.1.csv
│
├─ DE_rand_1_bin_F0.5_CR0.5/
└─ ... (27 configuraciones en total)
```
