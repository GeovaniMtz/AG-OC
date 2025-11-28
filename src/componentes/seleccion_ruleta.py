from random import Random
from typing import List, Sequence

def transformar_aptitud(costos: Sequence[float]) -> List[float]:
    """
    Transforma los valores de costo (minimización) en aptitud (maximización)
    utilizando escalamiento inverso.

    Aplica la transformación f(x) = 1 / (costo(x) + epsilon). Este método genera
    una alta presión selectiva hacia valores cercanos a cero, amplificando las
    diferencias entre soluciones muy buenas.

    Args:
        costos (Sequence[float]): Vector de costos a minimizar.

    Returns:
        List[float]: Vector de aptitudes normalizadas y positivas.
    """
    epsilon = 1e-6  # Constante de estabilidad numérica
    aptitudes = []
    for c in costos:
        # Corregir valores negativos
        costo_no_negativo = max(0.0, c)
        aptitudes.append(1.0 / (costo_no_negativo + epsilon))
    return aptitudes

def seleccion_ruleta(
    poblacion: List[List[int]], 
    aptitudes: Sequence[float], 
    k: int = 2, 
    rng: Random = None
) -> List[List[int]]:
    """
    Ejecuta el operador de selección proporcional a la aptitud (Roulette Wheel Selection).

    Selecciona k individuos de la población, donde la probabilidad de que un
    individuo sea elegido es proporcional a su valor de aptitud relativa.

    Args:
        poblacion (List[List[int]]): Conjunto de individuos candidatos.
        aptitudes (Sequence[float]): Valores de aptitud correspondientes a la población.
        k (int): Cantidad de individuos a seleccionar.
        rng (Random): Generador de números aleatorios.

    Returns:
        List[List[int]]: Lista de k individuos seleccionados.
    """
    if rng is None:
        raise ValueError("Se debe proveer un generador 'rng'")
        
    if len(poblacion) != len(aptitudes):
        raise ValueError("La población y las aptitudes deben tener la misma longitud.")
    if not all(a >= 0 for a in aptitudes):
        raise ValueError("Todas las aptitudes deben ser no-negativas para la ruleta.")

    total_aptitud = sum(aptitudes)
    
    # Caso base, sin aptitud
    if total_aptitud == 0:
        # Selección uniforme como mecanismo de fallback
        seleccionados = [rng.choice(poblacion) for _ in range(k)]
    else:
        # Muestreo estocástico ponderado por aptitud (ruleta)
        seleccionados = rng.choices(
            population=poblacion,
            weights=aptitudes,
            k=k
        )
    
    # Return de copias para evitar modificar la población original
    return [ind.copy() for ind in seleccionados]