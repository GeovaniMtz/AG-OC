from random import Random
from typing import List, Tuple

def cruza_un_punto(
    padre1: List[float],
    padre2: List[float],
    prob_cruza: float = 0.8,
    rng: Random = None
) -> Tuple[List[float], List[float]]:

    """
    Implementa el operador de cruza de un solo punto para codificación real.

    Genera descendencia explorando un intervalo extendido definido por la distancia
    entre los padres y el parámetro alpha, permitiendo tanto explotación como
    exploración.

    Args:
        padre1 (List[float]): Vector de variables de decisión del primer progenitor.
        padre2 (List[float]): Vector de variables de decisión del segundo progenitor.
        prob_cruza (float): Probabilidad de aplicar el operador [0, 1].
        rng (Random): Generador de números aleatorios para reproducibilidad.

    Returns:
        Tuple[List[float], List[float]]: Tupla con los dos descendientes generados.
    """

    if rng is None:
        raise ValueError("Se debe dar un generador 'rng'")

    n = len(padre1)
    if len(padre2) != n:
        raise ValueError("Los padres deben tener la misma longitud.")

    # Verificar si ocurre la cruza
    if rng.random() >= prob_cruza:
        return padre1.copy(), padre2.copy()

    # Si solo hay un gen, no tiene sentido cortar
    if n == 1:
        return padre1.copy(), padre2.copy()

    # Elegir un punto de corte (entre 1 y n-1)
    punto_corte = rng.randint(1, n - 1)

    # Crear hijos intercambiando sufijos
    hijo1 = padre1[:punto_corte] + padre2[punto_corte:]
    hijo2 = padre2[:punto_corte] + padre1[punto_corte:]

    return hijo1, hijo2
