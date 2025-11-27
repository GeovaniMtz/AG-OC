from random import Random
from typing import List, Tuple, Optional

def cruza_blx(
    padre1: List[float],
    padre2: List[float],
    prob_cruza: float = 0.9,
    alpha: float = 0.5,
    rng: Random = None,
    limite_inf: Optional[float] = None,  # <--- Nuevo argumento
    limite_sup: Optional[float] = None   # <--- Nuevo argumento
) -> Tuple[List[float], List[float]]:
    """
    Cruza BLX-α para representación real.
    - padre1, padre2: vectores de reales (misma longitud).
    - prob_cruza: probabilidad de aplicar cruza.
    - alpha: parámetro α de BLX.
    - limite_inf, limite_sup: límites del dominio para clipping (opcional).
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

    for x, y in zip(padre1, padre2):
        c_min = min(x, y)
        c_max = max(x, y)
        I = c_max - c_min

        low = c_min - alpha * I
        high = c_max + alpha * I

        # Dos hijos independientes dentro del mismo intervalo
        h1 = rng.uniform(low, high)
        h2 = rng.uniform(low, high)

        # --- REPARACIÓN DE LÍMITES (CLIPPING) ---
        # Si nos pasaron límites, nos aseguramos de que h1 y h2 no se salgan del rango
        if limite_inf is not None and limite_sup is not None:
            h1 = max(limite_inf, min(h1, limite_sup))
            h2 = max(limite_inf, min(h2, limite_sup))

        hijo1.append(h1)
        hijo2.append(h2)

    return hijo1, hijo2
    