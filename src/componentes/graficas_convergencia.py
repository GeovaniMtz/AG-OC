import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# CONFIGURACIÓN DEL SCRIPT
# ============================================================

# Ruta del archivo CSV con las curvas de evolución (una fila por generación)
RUTA_CSV_CURVAS = "resultados_ga_sphere_rastrigin_rosenbrock_curvas.csv"

funciones = ["sphere", "rastrigin", "rosenbrock"]
tipos_cruza = ["un_punto", "uniforme", "blx", "sbx"]

# Paleta de colores consistente con las otras gráficas
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
# GENERACIÓN DE GRÁFICAS DE CONVERGENCIA
# ============================================================

print("\n[INFO] Generando gráficas de convergencia (Escala Logarítmica)...\n")

for func in funciones:
    # Filtrar datos por función
    df_f = df[df["funcion"] == func]
    
    if df_f.empty:
        print(f"[WARN] No existen datos para la función: {func}")
        continue
    
    # Configuración de la figura
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    
    # Graficar una línea por cada operador
    for cruza in tipos_cruza:
        df_fc = df_f[df_f["tipo_cruza"] == cruza]
        
        if df_fc.empty:
            print(f"[WARN] Faltan datos para la combinación: {func} + {cruza}")
            continue
        
        # Calcular el promedio de 'mejor_generacion' agrupando por número de generación
        # Esto suaviza el ruido de las 30 repeticiones y muestra la tendencia central
        curva_prom = (
            df_fc.groupby("generacion")["mejor_generacion"]
                .mean()
                .sort_index()
        )
        
        # Trazar la línea
        ax.plot(
            curva_prom.index, 
            curva_prom.values, 
            label=cruza, 
            linewidth=2.5, 
            color=colores[cruza], 
            alpha=0.8
        )
    
    # --- Configuración Visual ---
    
    # Etiquetas y Título
    ax.set_xlabel("Generación", fontsize=12, fontweight='bold')
    ax.set_ylabel("Mejor Fitness Promedio (Log Scale)", fontsize=12, fontweight='bold')
    ax.set_title(f"Convergencia - Función {func.upper()}", 
                fontsize=14, fontweight='bold', pad=20)
    
    # Escala Logarítmica: VITAL para apreciar la precisión de BLX (10^-10)
    ax.set_yscale("log")
    
    # Grid y Leyenda
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.7, which="both")
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95, 
             edgecolor='black', fancybox=True, shadow=True)
    
    plt.tight_layout()
    
    # Guardado
    nombre_archivo = f"convergencia_{func}.png"
    plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
    print(f"[OK] Gráfica guardada: {nombre_archivo}")
    
    plt.close()

print("\n[INFO] Proceso finalizado.")