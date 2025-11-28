import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# CONFIGURACIÓN DEL SCRIPT
# ============================================================

# Ruta del archivo CSV con los resultados resumidos (una fila por ejecución)
RUTA_CSV_RESUMEN = "resultados_ga_sphere_rastrigin_rosenbrock.csv"

# Parámetros de análisis
funciones = ["sphere", "rastrigin", "rosenbrock"]
tipos_cruza = ["un_punto", "uniforme", "blx", "sbx"]

# Definición de paleta de colores para consistencia visual
colores = {
    "un_punto": "#FF6B6B",    # Rojo
    "uniforme": "#4ECDC4",    # Turquesa
    "blx": "#45B7D1",         # Azul
    "sbx": "#FFA07A"          # Salmón
}

# ============================================================
# CARGA Y VALIDACIÓN DE DATOS
# ============================================================

try:
    df = pd.read_csv(RUTA_CSV_RESUMEN)
    print(f"[INFO] Datos cargados exitosamente: {len(df)} filas.")
except FileNotFoundError:
    print(f"[ERROR] No se encontró el archivo: {RUTA_CSV_RESUMEN}")
    raise SystemExit(1)

# Detección dinámica de la columna de tiempo de ejecución
col_tiempo = None
posibles_nombres = ["tiempo_total_seg", "tiempo_total", "tiempo"]

for nombre in posibles_nombres:
    if nombre in df.columns:
        col_tiempo = nombre
        break

if col_tiempo is None:
    print("[ERROR] No se encontró una columna de tiempo válida.")
    print(f"        Columnas disponibles: {df.columns.tolist()}")
    raise SystemExit(1)

print(f"[INFO] Utilizando columna de tiempo: '{col_tiempo}'")

# ============================================================
# GENERACIÓN DE GRÁFICAS DE COSTO COMPUTACIONAL
# ============================================================

print("\n[INFO] Generando gráficas de tiempo de ejecución...\n")

for func in funciones:
    # Filtrar datos por la función objetivo actual
    df_f = df[df["funcion"] == func]
    
    if df_f.empty:
        print(f"[WARN] No existen datos para la función: {func}")
        continue
    
    # Cálculo de estadísticas descriptivas
    promedios = []
    desviaciones = []
    colores_barras = []
    
    for cruza in tipos_cruza:
        datos_cruza = df_f[df_f["tipo_cruza"] == cruza][col_tiempo]
        
        if datos_cruza.empty:
            promedios.append(0.0)
            desviaciones.append(0.0)
        else:
            promedios.append(datos_cruza.mean())
            desviaciones.append(datos_cruza.std())
        
        colores_barras.append(colores[cruza])
    
    # Configuración de la figura
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    
    x_pos = np.arange(len(tipos_cruza))
    
    # Renderizado de barras con barras de error (desviación estándar)
    barras = ax.bar(
        x_pos, 
        promedios, 
        yerr=desviaciones, 
        align='center', 
        alpha=0.9, 
        color=colores_barras,
        ecolor='black', 
        capsize=10
    )
    
    # Anotación de valores exactos sobre cada barra
    for barra in barras:
        height = barra.get_height()
        ax.annotate(
            f'{height:.4f}s',
            xy=(barra.get_x() + barra.get_width() / 2, height),
            xytext=(0, 3),  # Desplazamiento vertical de 3 puntos
            textcoords="offset points",
            ha='center', 
            va='bottom', 
            fontsize=10, 
            fontweight='bold'
        )

    # Configuración de ejes y estilos
    ax.set_xticks(x_pos)
    ax.set_xticklabels(tipos_cruza, fontsize=11, fontweight='bold')
    ax.set_ylabel("Tiempo Promedio (segundos)", fontsize=12, fontweight='bold')
    ax.set_title(f"Costo Computacional - Función {func.upper()}", 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Grid horizontal para facilitar lectura
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    # Ajuste dinámico del límite Y para acomodar las etiquetas superiores
    if promedios:
        ymax = max(promedios) * 1.15
        ax.set_ylim(0, ymax)

    plt.tight_layout()
    
    # Guardado de la gráfica
    nombre_archivo = f"tiempo_{func}.png"
    plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
    print(f"[OK] Gráfica guardada: {nombre_archivo}")
    
    plt.close()

print("\n[INFO] Proceso finalizado.")