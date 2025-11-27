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
    Cruza SBX (Simulated Binary Crossover) para representación real.
    Implementación estándar (sin truco de prob. extra por gen).

    - padre1, padre2: vectores de reales (misma longitud).
    - prob_cruza: probabilidad de aplicar la cruza SBX al individuo.
    - eta_c: índice de distribución.
    - limite_inf, limite_sup: cotas opcionales para recorte de los hijos.
    """
    if rng is None:
        raise ValueError("Se debe dar un generador 'rng'")

    n = len(padre1)
    if len(padre2) != n:
        raise ValueError("Los padres deben tener la misma longitud.")

    # Si no hay cruza: copiar padres tal cual
    if rng.random() >= prob_cruza:
        return padre1.copy(), padre2.copy()

    hijo1: List[float] = []
    hijo2: List[float] = []

    eps = 1e-14

    for x1, x2 in zip(padre1, padre2):
        # Si son prácticamente iguales, solo copiar
        if abs(x1 - x2) <= eps:
            c1, c2 = x1, x2
        else:
            # Asegurar x1 <= x2
            if x1 > x2:
                x1, x2 = x2, x1

            u = rng.random()
            if u <= 0.5:
                beta_q = (2.0 * u) ** (1.0 / (eta_c + 1.0))
            else:
                beta_q = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta_c + 1.0))

            c1 = 0.5 * ((x1 + x2) - beta_q * (x2 - x1))
            c2 = 0.5 * ((x1 + x2) + beta_q * (x2 - x1))

        # Reparación opcional a [limite_inf, limite_sup]
        if limite_inf is not None and limite_sup is not None:
            c1 = max(limite_inf, min(c1, limite_sup))
            c2 = max(limite_inf, min(c2, limite_sup))

        hijo1.append(c1)
        hijo2.append(c2)

    return hijo1, hijo2
