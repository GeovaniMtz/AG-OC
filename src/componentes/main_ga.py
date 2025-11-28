import random
import time
import csv
import sys

from typing import Callable, Dict, Tuple, List

from funciones import sphere, ackley, griewank, rastrigin, rosenbrock
from seleccion_ruleta import transformar_aptitud, seleccion_ruleta
from calcular_diversidad import calcular_diversidad
from cruza_un_punto import cruza_un_punto
from cruza_uniforme import cruza_uniforme
from cruza_blx import cruza_blx
from cruza_sbx import cruza_sbx
from mutacion_real import mutacion_real
from reemplazo_peores import reemplazo_peores

# =========================================
# 1. Configuración de Benchmarks
# =========================================

# Mapeo de nombres de funciones a sus implementaciones y límites de dominio
MAPA_FUNCIONES: Dict[str, Tuple[Callable[[List[float]], float], Tuple[float, float]]] = {
    "sphere":     (sphere,     (-5.12,   5.12)),
    "rastrigin":  (rastrigin,  (-5.12,   5.12)),
    "ackley":     (ackley,     (-30.0,   30.0)),
    "griewank":   (griewank,   (-600.0,  600.0)),
    "rosenbrock": (rosenbrock, (-2.048,  2.048)),
}

# =========================================
# 2. Funciones Auxiliares del AG
# =========================================

def inicializar_poblacion_reales(
    tam_pob: int,
    dim: int,
    a: float,
    b: float,
    rng: random.Random
) -> List[List[float]]:
    """
    Genera la población inicial con distribución uniforme dentro de los límites [a, b].
    """
    poblacion: List[List[float]] = []
    for _ in range(tam_pob):
        ind = [rng.uniform(a, b) for _ in range(dim)]
        poblacion.append(ind)
    return poblacion


def evaluar_poblacion(
    poblacion: List[List[float]],
    f: Callable[[List[float]], float]
) -> List[float]:
    """Calcula el costo (fitness) de cada individuo. Contexto de minimización."""
    return [f(ind) for ind in poblacion]


def crear_hijos_reales(
    p1: List[float],
    p2: List[float],
    pc: float,
    pm_gen: float,
    a: float,
    b: float,
    tipo_cruza: str,
    rng: random.Random,
    alpha_blx: float = 0.5,
    eta_c_sbx: float = 10.0,
    amplitud_mut: float = 0.1,
) -> Tuple[List[float], List[float]]:
    """
    Gestiona la reproducción: selecciona el operador de cruza y aplica mutación.
    
    Args:
        tipo_cruza: Identificador del operador ('un_punto', 'uniforme', 'blx', 'sbx').
        pm_gen: Probabilidad de mutación por gen.
        amplitud_mut: Intensidad de la mutación real.
    """
    tipo = tipo_cruza.lower()

    # Selección del operador de recombinación
    if tipo == "un_punto":
        c1, c2 = cruza_un_punto(p1, p2, prob_cruza=pc, rng=rng)
    elif tipo == "uniforme":
        c1, c2 = cruza_uniforme(p1, p2, prob_cruza=pc, rng=rng)
    elif tipo == "blx":
        c1, c2 = cruza_blx(p1, p2, prob_cruza=pc, alpha=alpha_blx, rng=rng, 
                           limite_inf=a, limite_sup=b)
    elif tipo == "sbx":
        c1, c2 = cruza_sbx(p1, p2, prob_cruza=pc, eta_c=eta_c_sbx, rng=rng, 
                           limite_inf=a, limite_sup=b)
    else:
        raise ValueError(f"Operador de cruza no reconocido: {tipo_cruza}")

    # Aplicación de mutación gaussiana a nivel de gen
    c1 = mutacion_real(c1, prob_mutacion_gen=pm_gen, a=a, b=b,
                       amplitud=amplitud_mut, rng=rng)
    c2 = mutacion_real(c2, prob_mutacion_gen=pm_gen, a=a, b=b,
                       amplitud=amplitud_mut, rng=rng)

    return c1, c2

# =========================================
# 3. Motor del Algoritmo Genético
# =========================================

def ejecutar_ga_real(
    nombre_func: str,
    dim: int = 10,
    tam_pob: int = 50,
    generaciones: int = 1000,
    pc: float = 0.9,
    tipo_cruza: str = "un_punto",
    porcentaje_reemplazo: float = 1.0,
    elitismo: int = 1,
    semilla: int = 42,
    alpha_blx: float = 0.5,
    eta_c_sbx: float = 10.0,
    amplitud_mut: float = 0.1,
) -> dict:
    """
    Ejecuta una instancia completa del AG. 
    Retorna métricas de desempeño y series de tiempo de la evolución.
    """
    if nombre_func not in MAPA_FUNCIONES:
        raise ValueError(f"Benchmark desconocido: {nombre_func}")

    # Inicialización de generador determinístico
    rng = random.Random(semilla)

    f, (a, b) = MAPA_FUNCIONES[nombre_func]

    # Heurística: Probabilidad de mutación inversamente proporcional a la dimensión
    pm_gen = 1.0 / dim

    # Inicialización y evaluación base
    poblacion = inicializar_poblacion_reales(tam_pob, dim, a, b, rng)
    costos = evaluar_poblacion(poblacion, f)

    # Estructuras para traza histórica
    curva_mejor: List[float] = []
    curva_promedio: List[float] = []
    curva_diversidad: List[float] = []

    t0 = time.perf_counter()

    for g in range(generaciones):
        # Selección de padres (Aptitud transformada para maximización)
        aptitudes = transformar_aptitud(costos)
        padres = seleccion_ruleta(poblacion, aptitudes, tam_pob, rng)

        # Ciclo de reproducción
        hijos: List[List[float]] = []
        for i in range(0, tam_pob, 2):
            p1 = padres[i]
            p2 = padres[(i + 1) % tam_pob] # Wrap-around para población impar

            h1, h2 = crear_hijos_reales(
                p1, p2,
                pc=pc,
                pm_gen=pm_gen,
                a=a, b=b,
                tipo_cruza=tipo_cruza,
                rng=rng,
                alpha_blx=alpha_blx,
                eta_c_sbx=eta_c_sbx,
                amplitud_mut=amplitud_mut,
            )
            hijos.append(h1)
            hijos.append(h2)

        # Recorte de excedentes
        hijos = hijos[:tam_pob]

        # Evaluación de descendencia
        costos_hijos = evaluar_poblacion(hijos, f)

        # Estrategia de reemplazo (Elitismo + Sustitución de peores)
        poblacion, costos = reemplazo_peores(
            poblacion=poblacion,
            hijos=hijos,
            apt_pob=costos,
            apt_hijos=costos_hijos,
            porcentaje=porcentaje_reemplazo,
            elitismo=elitismo
        )

        # Registro de métricas generacionales
        mejor = min(costos)
        promedio = sum(costos) / len(costos)
        diversidad = calcular_diversidad(poblacion)

        curva_mejor.append(mejor)
        curva_promedio.append(promedio)
        curva_diversidad.append(diversidad)

    t1 = time.perf_counter()
    tiempo_total = t1 - t0

    # Estadísticas finales
    mejor_final = min(costos)
    peor_final = max(costos)
    promedio_final = sum(costos) / len(costos)

    return {
        "nombre_func": nombre_func,
        "dim": dim,
        "tam_pob": tam_pob,
        "generaciones": generaciones,
        "pc": pc,
        "tipo_cruza": tipo_cruza,
        "porcentaje_reemplazo": porcentaje_reemplazo,
        "elitismo": elitismo,
        "semilla": semilla,
        "alpha_blx": alpha_blx,
        "eta_c_sbx": eta_c_sbx,
        "amplitud_mut": amplitud_mut,
        "mejor_final": mejor_final,
        "peor_final": peor_final,
        "promedio_final": promedio_final,
        "curva_mejor": curva_mejor,
        "curva_promedio": curva_promedio,
        "curva_diversidad": curva_diversidad,
        "poblacion_final": poblacion,
        "costos_finales": costos,
        "tiempo_total": tiempo_total,
    }

# =========================================
# 4. Ejecución de Experimentos
# =========================================

def correr_experimentos(
    nombre_archivo: str = "resultados_ga.csv",
    funciones: List[str] = None,
    cruzas: List[str] = None,
    dim: int = 10,
    tam_pob: int = 50,
    generaciones: int = 500,
    repeticiones: int = 20,
    modo_semillas: str = "independientes",
    base_semilla: int | None = None,
):
    """
    Orquesta la ejecución de múltiples corridas experimentales.
    Genera dos archivos CSV: uno con estadísticas finales y otro con la traza generacional completa.
    """

    if funciones is None:
        funciones = ["sphere", "rastrigin", "rosenbrock"]

    if cruzas is None:
        cruzas = ["un_punto", "uniforme", "blx", "sbx"]

    # Definición de nombres para archivos de salida
    nombre_curvas = nombre_archivo.replace(".csv", "_curvas.csv")

    with open(nombre_archivo, mode="w", newline="") as f_res, \
         open(nombre_curvas, mode="w", newline="") as f_curv:

        writer_res = csv.writer(f_res)
        writer_curv = csv.writer(f_curv)

        # Encabezados: Archivo de Resumen
        writer_res.writerow([
            "funcion",
            "tipo_cruza",
            "dim",
            "tam_pob",
            "generaciones",
            "repeticion",
            "semilla",
            "mejor_final",
            "peor_final",
            "promedio_final",
            "tiempo_total_seg",
            "diversidad",
        ])

        # Encabezados: Archivo de Curvas
        writer_curv.writerow([
            "funcion",
            "tipo_cruza",
            "dim",
            "tam_pob",
            "generaciones",
            "repeticion",
            "semilla",
            "generacion",
            "mejor_generacion",
            "promedio_generacion",
            "diversidad"
        ])

        # === Ejecución con semillas independientes ===
        if modo_semillas == "independientes":
            rep_global = 0
            for nombre_func in funciones:
                for tipo_cruza in cruzas:
                    for rep in range(repeticiones):
                        # Semilla única derivada del índice global para evitar colisiones
                        semilla = 1000 * rep_global + 123
                        rep_global += 1

                        print(f"[INFO] Función={nombre_func}, cruza={tipo_cruza}, "
                              f"rep={rep+1}/{repeticiones}, semilla={semilla}")

                        resultado = ejecutar_ga_real(
                            nombre_func=nombre_func,
                            dim=dim,
                            tam_pob=tam_pob,
                            generaciones=generaciones,
                            pc=0.9,
                            tipo_cruza=tipo_cruza,
                            porcentaje_reemplazo=1.0,
                            elitismo=1,
                            semilla=semilla,
                            alpha_blx=0.5,
                            eta_c_sbx=10.0,
                            amplitud_mut=0.1,
                        )

                        # Escritura de resumen
                        writer_res.writerow([
                            resultado["nombre_func"],
                            resultado["tipo_cruza"],
                            resultado["dim"],
                            resultado["tam_pob"],
                            resultado["generaciones"],
                            rep,
                            resultado["semilla"],
                            resultado["mejor_final"],
                            resultado["peor_final"],
                            resultado["promedio_final"],
                            resultado["tiempo_total"],
                        ])

                        # Escritura de curvas detalladas
                        curva_mejor = resultado["curva_mejor"]
                        curva_prom = resultado["curva_promedio"]
                        curva_div = resultado["curva_diversidad"]

                        for gen, (mejor_g, prom_g, div_g) in enumerate(
                            zip(curva_mejor, curva_prom, curva_div)
                        ):
                            writer_curv.writerow([
                                resultado["nombre_func"],
                                resultado["tipo_cruza"],
                                resultado["dim"],
                                resultado["tam_pob"],
                                resultado["generaciones"],
                                rep,
                                resultado["semilla"],
                                gen,
                                mejor_g,
                                prom_g,
                                div_g,
                            ])

        # === Ejecución con semillas por bloques (secuencial) ===
        elif modo_semillas == "bloques":
            if base_semilla is None:
                base_semilla = 42

            for rep in range(repeticiones):
                semilla = base_semilla + rep

                for nombre_func in funciones:
                    for tipo_cruza in cruzas:
                        print(f"[INFO] Función={nombre_func}, cruza={tipo_cruza}, "
                              f"rep={rep+1}/{repeticiones}, semilla={semilla}")

                        resultado = ejecutar_ga_real(
                            nombre_func=nombre_func,
                            dim=dim,
                            tam_pob=tam_pob,
                            generaciones=generaciones,
                            pc=0.9,
                            tipo_cruza=tipo_cruza,
                            porcentaje_reemplazo=1.0,
                            elitismo=1,
                            semilla=semilla,
                            alpha_blx=0.5,
                            eta_c_sbx=10.0,
                            amplitud_mut=0.1,
                        )

                        # Escritura de resumen
                        diversidad_final = resultado["curva_diversidad"][-1]

                        writer_res.writerow([
                            resultado["nombre_func"],
                            resultado["tipo_cruza"],
                            resultado["dim"],
                            resultado["tam_pob"],
                            resultado["generaciones"],
                            rep,
                            resultado["semilla"],
                            resultado["mejor_final"],
                            resultado["peor_final"],
                            resultado["promedio_final"],
                            resultado["tiempo_total"],
                            diversidad_final,
                        ])

                        # Escritura de curvas detalladas
                        curva_mejor = resultado["curva_mejor"]
                        curva_prom = resultado["curva_promedio"]
                        curva_diversidad = resultado["curva_diversidad"]
                        
                        for gen, (mejor_g, prom_g, div_g) in enumerate(
                            zip(curva_mejor, curva_prom, curva_diversidad)
                        ):
                            writer_curv.writerow([
                                resultado["nombre_func"],
                                resultado["tipo_cruza"],
                                resultado["dim"],
                                resultado["tam_pob"],
                                resultado["generaciones"],
                                rep,
                                resultado["semilla"],
                                gen,
                                mejor_g,
                                prom_g,
                                div_g,
                            ])
        else:
            raise ValueError(f"Modo de semillas no válido: {modo_semillas}")

    print(f"\n[OK] Resumen guardado en: {nombre_archivo}")
    print(f"[OK] Curvas guardadas en: {nombre_curvas}")


# =========================================
# 5. Punto de Entrada (CLI)
# =========================================

if __name__ == "__main__":
    # Gestión básica de argumentos para control de semillas
    # Uso: python main_ga.py [-s SEED]
    args = sys.argv[1:]
    modo_semillas = "independientes"
    base_semilla = None

    if len(args) >= 2 and args[0] in ("-s", "--seed"):
        try:
            base_semilla = int(args[1])
            modo_semillas = "bloques"
            print(f"[INFO] Modo de semillas 'bloques' activo. Base: {base_semilla}")
        except ValueError:
            print(f"[WARN] Semilla inválida '{args[1]}', revirtiendo a modo 'independientes'.")

    # Inicio de la batería de experimentos
    correr_experimentos(
        nombre_archivo="resultados_ga_sphere_rastrigin_rosenbrock.csv",
        funciones=["sphere", "rastrigin", "rosenbrock"],
        cruzas=["un_punto", "uniforme", "blx", "sbx"],
        dim=10,
        tam_pob=100,
        generaciones=1000,
        repeticiones=30,
        modo_semillas=modo_semillas,
        base_semilla=base_semilla,
    )