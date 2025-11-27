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
# 1. Mapa de funciones de prueba y dominios
# =========================================

MAPA_FUNCIONES: Dict[str, Tuple[Callable[[List[float]], float], Tuple[float, float]]] = {
    "sphere":     (sphere,     (-5.12,   5.12)),
    "rastrigin":  (rastrigin,  (-5.12,   5.12)),
    "ackley":     (ackley,     (-30.0,   30.0)),
    "griewank":   (griewank,   (-600.0,  600.0)),
    "rosenbrock": (rosenbrock, (-2.048,  2.048)),
}

# =========================================
# 2. Utilidades básicas para el AG en reales
# =========================================

def inicializar_poblacion_reales(
    tam_pob: int,
    dim: int,
    a: float,
    b: float,
    rng: random.Random
) -> List[List[float]]:
    """Crea una población inicial de vectores reales en [a, b]."""
    poblacion: List[List[float]] = []
    for _ in range(tam_pob):
        ind = [rng.uniform(a, b) for _ in range(dim)]
        poblacion.append(ind)
    return poblacion


def evaluar_poblacion(
    poblacion: List[List[float]],
    f: Callable[[List[float]], float]
) -> List[float]:
    """Evalúa la función objetivo en todos los individuos (MINIMIZACIÓN)."""
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
    Aplica la cruza elegida (un_punto, uniforme, blx, sbx) + mutación real
    para generar dos hijos a partir de dos padres.
    """
    tipo = tipo_cruza.lower()

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
        raise ValueError(f"tipo_cruza no soportado: {tipo_cruza}")

    # Mutación real gen a gen
    c1 = mutacion_real(c1, prob_mutacion_gen=pm_gen, a=a, b=b,
                       amplitud=amplitud_mut, rng=rng)
    c2 = mutacion_real(c2, prob_mutacion_gen=pm_gen, a=a, b=b,
                       amplitud=amplitud_mut, rng=rng)

    return c1, c2

# =========================================
# 3. AG principal (una sola corrida)
# =========================================

def ejecutar_ga_real(
    nombre_func: str,
    dim: int = 10,
    tam_pob: int = 50,
    generaciones: int = 1000,
    pc: float = 0.9,
    tipo_cruza: str = "un_punto",  # "un_punto", "uniforme", "blx", "sbx"
    porcentaje_reemplazo: float = 1.0,
    elitismo: int = 1,
    semilla: int = 42,
    alpha_blx: float = 0.5,
    eta_c_sbx: float = 10.0,
    amplitud_mut: float = 0.1,
) -> dict:
    """
    Ejecuta una corrida de AG con representación real para una función de prueba.
    Devuelve un diccionario con métricas finales, curvas e información de tiempo.
    """
    if nombre_func not in MAPA_FUNCIONES:
        raise ValueError(f"Función desconocida: {nombre_func}")

    # Generador de números aleatorios
    rng = random.Random(semilla)

    # Función objetivo y dominio
    f, (a, b) = MAPA_FUNCIONES[nombre_func]

    # Probabilidad de mutación por gen (heurística típica)
    pm_gen = 1.0 / dim

    # Población inicial
    poblacion = inicializar_poblacion_reales(tam_pob, dim, a, b, rng)

    # Costos iniciales
    costos = evaluar_poblacion(poblacion, f)

    # Curvas de evolución
    curva_mejor: List[float] = []
    curva_promedio: List[float] = []
    curva_diversidad: List[float] = []

    t0 = time.perf_counter()

    for g in range(generaciones):
        # --- Selección (usa costos -> aptitudes) ---
        aptitudes = transformar_aptitud(costos)  # convierte minimización a "fitness"
        padres = seleccion_ruleta(poblacion, aptitudes, tam_pob, rng)

        # --- Cruza + mutación: generamos una camada de hijos ---
        hijos: List[List[float]] = []
        for i in range(0, tam_pob, 2):
            p1 = padres[i]
            p2 = padres[(i + 1) % tam_pob]  # por si tam_pob es impar

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

        # Ajustar tamaño (por si se generó 1 hijo extra)
        hijos = hijos[:tam_pob]

        # --- Evaluar hijos (MINIMIZACIÓN) ---
        costos_hijos = evaluar_poblacion(hijos, f)

        # --- Reemplazo (usa costos como "aptitud" a minimizar) ---
        poblacion, costos = reemplazo_peores(
            poblacion=poblacion,
            hijos=hijos,
            apt_pob=costos,
            apt_hijos=costos_hijos,
            porcentaje=porcentaje_reemplazo,
            elitismo=elitismo
        )

        # --- Registrar métricas por generación ---
        mejor = min(costos)
        promedio = sum(costos) / len(costos)
        diversidad = calcular_diversidad(poblacion)

        curva_mejor.append(mejor)
        curva_promedio.append(promedio)
        curva_diversidad.append(diversidad)

    t1 = time.perf_counter()
    tiempo_total = t1 - t0

    # Métricas finales
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
# 4. Experimentos múltiples + CSV
# =========================================

def correr_experimentos(
    nombre_archivo: str = "resultados_ga.csv",
    funciones: List[str] = None,
    cruzas: List[str] = None,
    dim: int = 10,
    tam_pob: int = 50,
    generaciones: int = 500,
    repeticiones: int = 20,
    modo_semillas: str = "independientes",  # "independientes" o "bloques"
    base_semilla: int | None = None,
):
    """
    Corre varias veces el AG para distintas funciones y operadores de cruza,
    y guarda:
      - un CSV de resumen (una fila por corrida)
      - un CSV de curvas (una fila por generación)
    """

    if funciones is None:
        funciones = ["sphere", "rastrigin", "rosenbrock"]

    if cruzas is None:
        cruzas = ["un_punto", "uniforme", "blx", "sbx"]

    # Archivo principal (resumen) y archivo de curvas
    nombre_curvas = nombre_archivo.replace(".csv", "_curvas.csv")

    with open(nombre_archivo, mode="w", newline="") as f_res, \
         open(nombre_curvas, mode="w", newline="") as f_curv:

        writer_res = csv.writer(f_res)
        writer_curv = csv.writer(f_curv)

        # Encabezados del CSV de RESUMEN
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

        # Encabezados del CSV de CURVAS
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

        # =========================
        #  MODO SEMILLAS INDEPENDES
        # =========================
        if modo_semillas == "independientes":
            rep_global = 0
            for nombre_func in funciones:
                for tipo_cruza in cruzas:
                    for rep in range(repeticiones):
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

                        # ------- RESUMEN --------
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

                        # ------- CURVAS ---------
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

        # =====================
        #  MODO SEMILLAS BLOQUE
        # =====================
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

                        # ------- RESUMEN --------

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

                        # ------- CURVAS ---------
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
            raise ValueError(f"modo_semillas no reconocido: {modo_semillas}")

    print(f"\n[OK] Resultados RESUMEN en: {nombre_archivo}")
    print(f"[OK] Curvas por generación en: {nombre_curvas}")



# =========================================
# 5. Punto de entrada
# =========================================

if __name__ == "__main__":
    # Parseo MUY simple de argumentos:
    #   python main_ga.py           -> modo_semillas = "independientes"
    #   python main_ga.py -s 42     -> modo_semillas = "bloques", base_semilla = 42
    #   python main_ga.py --seed 10 -> igual que -s 10

    args = sys.argv[1:]
    modo_semillas = "independientes"
    base_semilla = None

    if len(args) >= 2 and args[0] in ("-s", "--seed"):
        try:
            base_semilla = int(args[1])
            modo_semillas = "bloques"
            print(f"[INFO] Usando modo de semillas 'bloques' con base = {base_semilla}")
        except ValueError:
            print(f"[WARN] Semilla base inválida '{args[1]}', usando modo 'independientes'.")

    # --- MODO EXPERIMENTOS (muchas corridas + CSV) ---
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

