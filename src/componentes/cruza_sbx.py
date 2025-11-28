from random import Random
from typing import List, Tuple, Optional

def cruza_sbx(
    padre1: List[float],
    padre2: List[float],
    prob_cruza: float = 0.9,
    eta_c: float = 10.0,
    rng: Random = None,
    limite_inf: Optional[float] = None,
    limite_sup: Optional[float] = None,
) -> Tuple[List[float], List[float]]:

    """
    Implementa el operador SBX (Simulated Binary Crossover) para codificación real.

    Simula la distribución de probabilidad de la cruza de un punto en representaciones
    binarias, adaptada a espacios continuos. Genera descendencia cercana a los padres o
    alejada de ellos según el índice de distribución (eta_c).

    Args:
        padre1 (List[float]): Vector de variables de decisión del primer progenitor.
        padre2 (List[float]): Vector de variables de decisión del segundo progenitor.
        prob_cruza (float): Probabilidad de aplicar el operador [0, 1].
        eta_c (float): Índice de distribución. Valores altos generan hijos semejantes
                       a los padres (explotación); valores bajos permiten mayor diversidad (exploración).
        rng (Random): Generador de números aleatorios.
        limite_inf (Optional[float]): Cota inferior para restringir el espacio de búsqueda.
        limite_sup (Optional[float]): Cota superior para restringir el espacio de búsqueda.

    Returns:
        Tuple[List[float], List[float]]: Tupla con los dos descendientes generados.
    """

    if rng is None:
        raise ValueError("Se debe dar un generador 'rng'")

    n = len(padre1)
    if len(padre2) != n:
        raise ValueError("Los padres deben tener la misma longitud.")

    # Copiar padres si no se cumple la probabilidad de cruza
    if rng.random() >= prob_cruza:
        return padre1.copy(), padre2.copy()

    hijo1: List[float] = []
    hijo2: List[float] = []

    eps = 1e-14

    for x1, x2 in zip(padre1, padre2):
        # Manejo de genes idénticos o numéricamente muy cercanos para evitar inestabilidad
        if abs(x1 - x2) <= eps:
            c1, c2 = x1, x2
        else:
            # Ordenamiento de variables para el cálculo (x1 < x2)
            if x1 > x2:
                x1, x2 = x2, x1

            u = rng.random()
            if u <= 0.5:
                beta_q = (2.0 * u) ** (1.0 / (eta_c + 1.0))
            else:
                beta_q = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta_c + 1.0))

            c1 = 0.5 * ((x1 + x2) - beta_q * (x2 - x1))
            c2 = 0.5 * ((x1 + x2) + beta_q * (x2 - x1))

        # Restricción de límites (clipping) para asegurar factibilidad
        if limite_inf is not None and limite_sup is not None:
            c1 = max(limite_inf, min(c1, limite_sup))
            c2 = max(limite_inf, min(c2, limite_sup))

        hijo1.append(c1)
        hijo2.append(c2)

    return hijo1, hijo2
    