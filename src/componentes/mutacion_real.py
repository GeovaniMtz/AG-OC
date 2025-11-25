from random import Random
from typing import List

def mutacion_real(
    individuo: List[float],
    prob_mutacion_gen: float = 0.1,
    a: float = -5.0,
    b: float = 5.0,
    amplitud: float = 0.1,
    rng: Random = None
) -> List[float]:
    """
    Mutación para representación real (minimización en [a, b]).

    - individuo: vector de números reales.
    - prob_mutacion_gen: probabilidad de mutar cada gen.
    - [a, b]: límites del dominio (se recorta el valor mutado).
    - amplitud: fracción del rango (b-a) usada como máximo cambio.
      Ejemplo: amplitud = 0.1 -> ruido en [-0.1*(b-a), 0.1*(b-a)].
    - rng: generador Random para reproducibilidad.

    Regresa un NUEVO individuo mutado.
    """
    if rng is None:
        raise ValueError("Se debe proporcionar un generador 'rng'")

    if not 0.0 <= prob_mutacion_gen <= 1.0:
        raise ValueError("prob_mutacion_gen debe estar en [0, 1]")

    # Copia para no modificar el original
    hijo = individuo.copy()
    rango = b - a
    max_cambio = amplitud * rango

    for i, valor in enumerate(hijo):
        if rng.random() < prob_mutacion_gen:
            # Ruido uniforme pequeño
            ruido = rng.uniform(-max_cambio, max_cambio)
            nuevo_valor = valor + ruido
            # Recortamos al dominio [a, b]
            if nuevo_valor < a:
                nuevo_valor = a
            elif nuevo_valor > b:
                nuevo_valor = b
            hijo[i] = nuevo_valor

    return hijo
