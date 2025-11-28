from random import Random
from typing import List, Tuple, Optional

def cruza_blx(
    padre1: List[float],
    padre2: List[float],
    prob_cruza: float = 0.9,
    alpha: float = 0.5,
    rng: Random = None,
    limite_inf: Optional[float] = None,
    limite_sup: Optional[float] = None
) -> Tuple[List[float], List[float]]:
    """
    Implementa el operador de cruza BLX-alpha (Blend Crossover) para codificación real.

    Genera descendencia explorando un intervalo extendido definido por la distancia
    entre los padres y el parámetro alpha, permitiendo tanto explotación como
    exploración.

    Args:
        padre1 (List[float]): Vector de variables de decisión del primer progenitor.
        padre2 (List[float]): Vector de variables de decisión del segundo progenitor.
        prob_cruza (float): Probabilidad de aplicar el operador [0, 1].
        alpha (float): Coeficiente de expansión del intervalo. Controla la exploración.
        rng (Random): Generador de números aleatorios para reproducibilidad.
        limite_inf (Optional[float]): Cota inferior para restringir el espacio de búsqueda.
        limite_sup (Optional[float]): Cota superior para restringir el espacio de búsqueda.

    Returns:
        Tuple[List[float], List[float]]: Tupla con los dos descendientes generados.
    """
    if rng is None:
        raise ValueError("Se debe proporcionar un generador 'rng'")

    n = len(padre1)
    if len(padre2) != n:
        raise ValueError("Los vectores padres deben tener la misma dimensión.")

    # Copiar padres si no se cumple la probabilidad de cruza
    if rng.random() >= prob_cruza:
        return padre1.copy(), padre2.copy()

    hijo1 = []
    hijo2 = []

    for x, y in zip(padre1, padre2):
        c_min = min(x, y)
        c_max = max(x, y)
        I = c_max - c_min

        # Definición del intervalo extendido [min - I*alpha, max + I*alpha]
        low = c_min - alpha * I
        high = c_max + alpha * I

        # Muestreo independiente para cada hijo
        h1 = rng.uniform(low, high)
        h2 = rng.uniform(low, high)

        # Restricción de límites (clipping) para asegurar factibilidad
        if limite_inf is not None and limite_sup is not None:
            h1 = max(limite_inf, min(h1, limite_sup))
            h2 = max(limite_inf, min(h2, limite_sup))

        hijo1.append(h1)
        hijo2.append(h2)

    return hijo1, hijo2
    