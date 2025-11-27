import numpy as np
from typing import List

def calcular_diversidad(poblacion: List[List[float]]) -> float:
    """
    Calcula la diversidad de una población usando desviación estándar promediada.
    
    Diversidad = promedio de las desviaciones estándar de cada dimensión
    
    Interpretación:
    - Diversidad ALTA (p. ej. 2.0): Población muy esparcida, buena exploración
    - Diversidad MEDIA (p. ej. 0.5): Población moderadamente diversa
    - Diversidad BAJA (p. ej. 0.01): Población convergida, poca exploración
    
    Args:
        poblacion: Lista de individuos, cada uno es una lista de reales
    
    Returns:
        Diversidad promedio (desviación estándar promediada entre dimensiones)
    """
    if not poblacion or len(poblacion) == 0:
        return 0.0
    
    # Convertir población a array numpy para cálculos
    pob_array = np.array(poblacion, dtype=float)
    
    # pob_array.shape = (num_individuos, num_dimensiones)
    # axis=0 → calcula std por cada dimensión
    desv_std_por_dimension = np.std(pob_array, axis=0)
    
    # Promediar las desv. estándar de todas las dimensiones
    diversidad = np.mean(desv_std_por_dimension)
    
    return float(diversidad)


# ============================================================
# ALTERNATIVA: Diversidad basada en distancia pairwise
# ============================================================

def calcular_diversidad_distancia(poblacion: List[List[float]]) -> float:
    """
    Calcula diversidad como distancia Euclidiana promedio entre individuos.
    
    Más costoso (O(n²)) pero más robusto que std.
    
    Args:
        poblacion: Lista de individuos
    
    Returns:
        Distancia promedio entre pares de individuos
    """
    if len(poblacion) <= 1:
        return 0.0
    
    pob_array = np.array(poblacion, dtype=float)
    n = len(pob_array)
    
    # Calcular todas las distancias Euclidianas
    suma_distancias = 0.0
    num_pares = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            distancia = np.linalg.norm(pob_array[i] - pob_array[j])
            suma_distancias += distancia
            num_pares += 1
    
    if num_pares == 0:
        return 0.0
    
    diversidad_media = suma_distancias / num_pares
    return float(diversidad_media)