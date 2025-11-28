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
    Implementa el operador de mutación uniforme para codificación real.

    Aplica una perturbación estocástica a cada variable de decisión con base en una
    probabilidad dada, restringiendo el resultado al dominio factible [a, b].

    Args:
        individuo (List[float]): Vector de variables de decisión a mutar.
        prob_mutacion_gen (float): Probabilidad de aplicar mutación a un gen específico [0, 1].
        a (float): Cota inferior del espacio de búsqueda.
        b (float): Cota superior del espacio de búsqueda.
        amplitud (float): Factor de escala que define el rango máximo de la perturbación
                          relativo al tamaño del dominio (b - a).
        rng (Random): Generador de números aleatorios.

    Returns:
        List[float]: Nuevo individuo con las mutaciones aplicadas.
    """
    if rng is None:
        raise ValueError("Se debe proporcionar un generador 'rng'")

    if not 0.0 <= prob_mutacion_gen <= 1.0:
        raise ValueError("prob_mutacion_gen debe estar en [0, 1]")

    # Generar copia para preservar el individuo original
    hijo = individuo.copy()
    rango = b - a
    max_cambio = amplitud * rango

    for i, valor in enumerate(hijo):
        if rng.random() < prob_mutacion_gen:
            # Aplicar ruido uniforme centrado en 0
            ruido = rng.uniform(-max_cambio, max_cambio)
            nuevo_valor = valor + ruido
            
            # Saturación (clipping) para respetar los límites del dominio
            if nuevo_valor < a:
                nuevo_valor = a
            elif nuevo_valor > b:
                nuevo_valor = b
            hijo[i] = nuevo_valor

    return hijo
    