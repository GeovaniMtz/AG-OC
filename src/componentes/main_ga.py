import random
import time
from typing import Callable, Dict, Tuple, List

from funciones import sphere, ackley, griewank, rastrigin, rosenbrock
from seleccion_ruleta import transformar_aptitud, seleccion_ruleta
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
        c1, c2 = cruza_blx(p1, p2, prob_cruza=pc, alpha=alpha_blx, rng=rng)
    elif tipo == "sbx":
        c1, c2 = cruza_sbx(p1, p2, prob_cruza=pc, eta_c=eta_c_sbx, rng=rng)
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
        curva_mejor.append(mejor)
        curva_promedio.append(promedio)

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
        "poblacion_final": poblacion,
        "costos_finales": costos,
        "tiempo_total": tiempo_total,
    }

# =========================================
# 4. Ejemplo de uso rápido
# =========================================

if __name__ == "__main__":
    # Ejemplo: una sola corrida en Sphere con BLX
    resultado = ejecutar_ga_real(
        nombre_func="sphere",
        dim=5,
        tam_pob=30,
        generaciones=200,
        pc=0.9,
        tipo_cruza="blx",       # "un_punto", "uniforme", "blx", "sbx"
        porcentaje_reemplazo=1.0,
        elitismo=1,
        semilla=123,
        alpha_blx=0.5,
        eta_c_sbx=10.0,
        amplitud_mut=0.1,
    )

    print(f"Función: {resultado['nombre_func']}")
    print(f"Tipo de cruza: {resultado['tipo_cruza']}")
    print("Mejor valor final:", resultado["mejor_final"])
    print("Tiempo total (s):", resultado["tiempo_total"])
