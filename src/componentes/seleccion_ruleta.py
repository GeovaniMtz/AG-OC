from random import Random
from typing import List, Sequence

def transformar_aptitud(costos: Sequence[float]) -> List[float]:
    """
    Transforma una lista de costos (minimización) en una lista de aptitudes
    (maximización) usando la fórmula 1 / (costo + epsilon).
    """
    epsilon = 1e-6  # Para evitar división por cero
    aptitudes = []
    for c in costos:
        # Asegurarnos de que el costo no sea negativo
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
    Selecciona 'k' individuos (padres) de la población usando el método
    de la ruleta, basado en la lista de 'aptitudes' (maximización).
    
    Requiere un generador 'rng' para ser reproducible.
    """
    if rng is None:
        raise ValueError("Se debe proveer un generador 'rng'")
        
    if len(poblacion) != len(aptitudes):
        raise ValueError("La población y las aptitudes deben tener la misma longitud.")
    if not all(a >= 0 for a in aptitudes):
        raise ValueError("Todas las aptitudes deben ser no-negativas para la ruleta.")

    total_aptitud = sum(aptitudes)
    
    # Si la aptitud total es cero (todos los individuos tienen aptitud 0),
    # seleccionamos al azar para evitar división por cero.
    if total_aptitud == 0:
        # Usa rng.choice
        seleccionados = [rng.choice(poblacion) for _ in range(k)]
    else:
        # Selección ponderada usando las aptitudes
        seleccionados = rng.choices(
            population=poblacion,
            weights=aptitudes,
            k=k
        )
    
    # Devolvemos copias para evitar modificar accidentalmente a los seleccionados
    return [ind.copy() for ind in seleccionados]