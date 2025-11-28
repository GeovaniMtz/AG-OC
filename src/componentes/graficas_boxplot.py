import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURACIÓN DEL SCRIPT
# ============================================================

# Ruta del archivo CSV con los resultados resumidos (una fila por ejecución)
RUTA_CSV_RESUMEN = "resultados_ga_sphere_rastrigin_rosenbrock.csv"

# Parámetros de análisis
funciones = ["sphere", "rastrigin", "rosenbrock"]
tipos_cruza = ["un_punto", "uniforme", "blx", "sbx"]

# Definición de paleta de colores por operador para consistencia visual
colores = {
    "un_punto": "#FF6B6B",    # Rojo
    "uniforme": "#4ECDC4",    # Turquesa
    "blx": "#45B7D1",         # Azul
    "sbx": "#FFA07A"          # Salmón
}

# ============================================================
# CARGA DE DATOS
# ============================================================

try:
    df = pd.read_csv(RUTA_CSV_RESUMEN)
    print(f"[INFO] Datos cargados exitosamente: {len(df)} registros.")
except FileNotFoundError:
    print(f"[ERROR] No se encontró el archivo: {RUTA_CSV_RESUMEN}")
    raise SystemExit(1)

# ============================================================
# GENERACIÓN DE DIAGRAMAS DE CAJA (BOXPLOTS)
# ============================================================

print("\n[INFO] Generando diagramas de caja para calidad final...\n")

for func in funciones:
    # Filtrar datos por la función objetivo actual
    df_f = df[df["funcion"] == func]
    
    if df_f.empty:
        print(f"[WARN] No existen datos para la función: {func}")
        continue
    
    # Estructuración de datos para matplotlib
    datos_por_cruza = []
    etiquetas = []
    colores_lista = []
    
    for cruza in tipos_cruza:
        # Extraer vector de 'mejor_final' para el operador actual
        valores = df_f[df_f["tipo_cruza"] == cruza]["mejor_final"].values
        
        datos_por_cruza.append(valores)
        etiquetas.append(cruza)
        colores_lista.append(colores[cruza])
    
    # Configuración de la figura
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    
    # Creación del Boxplot
    # patch_artist=True habilita el relleno de color en las cajas
    bplot = ax.boxplot(
        datos_por_cruza, 
        labels=etiquetas,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.5),
        flierprops=dict(marker='o', markerfacecolor='red', markersize=5, linestyle='none')
    )
    
    # Asignación de colores correspondientes a cada caja
    for patch, color in zip(bplot['boxes'], colores_lista):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    
    # Configuración de escala logarítmica en el eje Y
    # Esencial para visualizar diferencias de órdenes de magnitud en minimización
    ax.set_yscale("log")
    
    # Configuración de etiquetas y títulos
    ax.set_xlabel("Operador de Cruza", fontsize=12, fontweight='bold')
    ax.set_ylabel("Mejor Costo Final (Escala Log)", fontsize=12, fontweight='bold')
    ax.set_title(f"Distribución de Calidad Final - Función {func.upper()}", 
                fontsize=14, fontweight='bold', pad=20)
    
    # Configuración del grid (solo horizontal para facilitar lectura de niveles)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.7, axis='y', which='both')
    
    plt.tight_layout()
    
    # Guardado de la imagen
    nombre_archivo = f"boxplot_calidad_{func}.png"
    plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
    print(f"[OK] Gráfica guardada: {nombre_archivo}")
    
    plt.close()

print("\n[INFO] Proceso finalizado.")