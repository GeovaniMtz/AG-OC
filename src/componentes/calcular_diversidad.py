import numpy as np
from typing import List

def calcular_diversidad(poblacion: List[List[float]]) -> float:
    """
    Calcula la diversidad fenotípica de la población utilizando la desviación
    estándar promedio por dimensión.

    Args:
        poblacion (List[List[float]]): Matriz de individuos (población actual).

    Returns:
        float: Promedio de las desviaciones estándar de cada variable de decisión.
               Valores cercanos a 0 indican convergencia de la población.
    """
    if not poblacion:
        return 0.0
    
    pob_array = np.array(poblacion, dtype=float)
    
    # Calcular desviación estándar a lo largo de las dimensiones (axis=0)
    desv_std_por_dimension = np.std(pob_array, axis=0)
    
    return float(np.mean(desv_std_por_dimension))


def calcular_diversidad_distancia(poblacion: List[List[float]]) -> float:
    """
    Calcula la diversidad basada en la distancia Euclidiana promedio entre
    pares de individuos.
    
    Nota: Este método tiene complejidad O(N^2), por lo que se recomienda
    usarlo solo con poblaciones pequeñas o para análisis específicos.

    Args:
        poblacion (List[List[float]]): Lista de individuos.

    Returns:
        float: Distancia promedio entre todos los pares únicos de individuos.
    """
    n = len(poblacion)
    if n <= 1:
        return 0.0
    
    pob_array = np.array(poblacion, dtype=float)
    suma_distancias = 0.0
    num_pares = 0
    
    # Acumular distancias de pares únicos (i, j) donde j > i
    for i in range(n):
        for j in range(i + 1, n):
            distancia = np.linalg.norm(pob_array[i] - pob_array[j])
            suma_distancias += distancia
            num_pares += 1
    
    if num_pares == 0:
        return 0.0
    
    return float(suma_distancias / num_pares)