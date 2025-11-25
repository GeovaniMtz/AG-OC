from random import Random
from typing import List, Tuple

def cruza_sbx(
    padre1: List[float],
    padre2: List[float],
    prob_cruza: float = 0.9,
    eta_c: float = 10.0,
    rng: Random = None
) -> Tuple[List[float], List[float]]:
    """
    Cruza SBX (Simulated Binary Crossover) para representación real.
    - padre1, padre2: vectores de reales (misma longitud).
    - prob_cruza: probabilidad de aplicar cruza.
    - eta_c: índice de distribución.
    Nota: no usa límites [a,b] aquí; puedes recortar después si quieres.
    """
    if rng is None:
        raise ValueError("Se debe dar un generador 'rng'")

    n = len(padre1)
    if len(padre2) != n:
        raise ValueError("Los padres deben tener la misma longitud.")

    # Si no hay cruza: copiar padres
    if rng.random() >= prob_cruza:
        return padre1.copy(), padre2.copy()

    hijo1 = []
    hijo2 = []

    eps = 1e-14

    for x1, x2 in zip(padre1, padre2):
        if rng.random() <= 0.5 and abs(x1 - x2) > eps:
            if x1 > x2:
                x1, x2 = x2, x1

            u = rng.random()
            beta = 1.0 + (2.0 * (x1 - x2) / (x2 - x1))  # se cancela, pero dejo formula original
            # En la práctica, la beta clásica:
            if u <= 0.5:
                beta_q = (2.0 * u) ** (1.0 / (eta_c + 1.0))
            else:
                beta_q = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta_c + 1.0))

            c1 = 0.5 * ((x1 + x2) - beta_q * (x2 - x1))
            c2 = 0.5 * ((x1 + x2) + beta_q * (x2 - x1))
        else:
            # Sin cruza "efectiva", solo copiamos
            c1 = x1
            c2 = x2

        hijo1.append(c1)
        hijo2.append(c2)

    return hijo1, hijo2
