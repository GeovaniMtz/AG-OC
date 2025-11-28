import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURACIÓN DEL SCRIPT
# ============================================================

# Ruta del archivo CSV con las curvas de evolución (incluyendo diversidad)
RUTA_CSV_CURVAS = "resultados_ga_sphere_rastrigin_rosenbrock_curvas.csv"

# Parámetros de análisis
funciones = ["sphere", "rastrigin", "rosenbrock"]
tipos_cruza = ["un_punto", "uniforme", "blx", "sbx"]

# Definición de paleta de colores por operador
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
    df = pd.read_csv(RUTA_CSV_CURVAS)
    print(f"[INFO] Datos cargados exitosamente: {len(df)} filas.")
except FileNotFoundError:
    print(f"[ERROR] No se encontró el archivo: {RUTA_CSV_CURVAS}")
    raise SystemExit(1)

# ============================================================
# VERIFICACIÓN DE COLUMNAS
# ============================================================

# Detección automática del nombre de la columna de diversidad
columna_diversidad = None

if "diversidad" in df.columns:
    columna_diversidad = "diversidad"
    print(f"[INFO] Columna detectada: 'diversidad'")
elif "diversidad_generacion" in df.columns:
    columna_diversidad = "diversidad_generacion"
    print(f"[INFO] Columna detectada: 'diversidad_generacion'")
else:
    print(f"[ERROR] No se encontró ninguna columna de diversidad.")
    print(f"        Columnas disponibles: {df.columns.tolist()}")
    raise SystemExit(1)

# ============================================================
# GENERACIÓN DE GRÁFICAS DE DIVERSIDAD
# ============================================================

print("\n[INFO] Generando gráficas de diversidad...\n")

for func in funciones:
    # Filtrar datos por función
    df_f = df[df["funcion"] == func]
    
    if df_f.empty:
        print(f"[WARN] No existen datos para la función: {func}")
        continue
    
    # Configuración de la figura
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    
    for cruza in tipos_cruza:
        # Filtrar datos por operador
        df_fc = df_f[df_f["tipo_cruza"] == cruza]
        
        if df_fc.empty:
            print(f"[WARN] Faltan datos para la combinación: {func} + {cruza}")
            continue
        
        # Calcular el promedio de diversidad por generación
        curva_div = (
            df_fc.groupby("generacion")[columna_diversidad]
                 .mean()
                 .sort_index()
        )
        
        # Trazar la curva
        ax.plot(
            curva_div.index,
            curva_div.values,
            label=cruza,
            linewidth=2.5,
            color=colores[cruza],
            alpha=0.8,
            # Configuración opcional de marcadores
            # marker='o',
            # markersize=3,
            # markevery=max(1, len(curva_div) // 10)
        )
    
    # Configuración de escala logarítmica
    # Permite visualizar cambios de diversidad en órdenes de magnitud muy pequeños
    ax.set_yscale("log")
    
    # Configuración opcional de Zoom (para analizar fases iniciales)
    # ax.set_xlim(0, 100) 
    
    # Etiquetas y Títulos
    ax.set_xlabel("Generación", fontsize=12, fontweight='bold')
    ax.set_ylabel("Diversidad Promedio (Escala Log)", fontsize=12, fontweight='bold')
    ax.set_title(f"Pérdida de Diversidad - Función {func.upper()}",
                 fontsize=14, fontweight='bold', pad=20)
    
    # Configuración de Grid y Leyenda
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.7, which="both")
    ax.legend(loc='best', fontsize=11,
              framealpha=0.95, edgecolor='black',
              fancybox=True, shadow=True)
    
    plt.tight_layout()
    
    # Guardado de la gráfica
    nombre_archivo = f"diversidad_{func}.png"
    plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
    print(f"[OK] Gráfica guardada: {nombre_archivo}")
    
    plt.close()

print("\n[INFO] Proceso finalizado.")
print("\nArchivos generados:")
for func in funciones:
    print(f"  - diversidad_{func}.png")