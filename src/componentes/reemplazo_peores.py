from typing import List, Sequence, Tuple
import math

def reemplazo_peores(
    poblacion: List[List[float]],
    hijos: List[List[float]],
    apt_pob: Sequence[float],
    apt_hijos: Sequence[float],
    porcentaje: float = 1.0,
    elitismo: int = 1
) -> Tuple[List[List[float]], List[float]]:
    """
    Implementa el Reemplazo Generacional con Elitismo.

    Sustituye la población actual completa por los mejores descendientes generados,
    pero garantiza la supervivencia de los mejores individuos (élite) de la generación
    anterior si estos superan a los nuevos candidatos.

    Args:
        poblacion (List[List[float]]): Población actual (padres).
        hijos (List[List[float]]): Descendencia generada (pool de hijos).
        apt_pob (Sequence[float]): Valores de costo de la población actual (menor es mejor).
        apt_hijos (Sequence[float]): Valores de costo de la descendencia (menor es mejor).
        porcentaje (float): (No utilizado en esta versión simplificada, se asume 1.0).
        elitismo (int): Número de mejores padres que se garantiza preservar.

    Returns:
        Tuple[List[List[float]], List[float]]: Nueva población y sus costos asociados.
    """
    N = len(poblacion)
    
    # Validación básica de dimensiones
    if len(apt_pob) != N or len(hijos) != len(apt_hijos):
        raise ValueError("Dimensiones inconsistentes entre población y costos.")

    # Identificar jerarquía de padres (índices ordenados por mejor costo)
    # idx_padres[0] es el índice del mejor padre (el "campeón")
    idx_padres = sorted(range(N), key=lambda i: apt_pob[i])

    # 1. Selección de Sobrevivientes (Hijos)
    # Seleccionamos los N mejores hijos disponibles para formar la base de la nueva generación
    # Se ordenan por costo ascendente (mejores primero)
    idx_mej_hijos = sorted(range(len(hijos)), key=lambda i: apt_hijos[i])[:N]
    
    nueva_pob = [hijos[i][:] for i in idx_mej_hijos]
    nuevas_apt = [apt_hijos[i] for i in idx_mej_hijos]

    # 2. Aplicación de Elitismo
    # Si el mejor padre de la generación anterior es mejor que el peor hijo aceptado,
    # reemplazamos al peor hijo con ese padre élite para no perder calidad.
    e = max(0, min(elitismo, N))
    
    if e >= 1:
        # Encontramos al peor individuo de la nueva población (el candidato a salir)
        idx_peor_nueva = max(range(N), key=lambda i: nuevas_apt[i])
        
        # Recuperamos al mejor padre absoluto
        best_padre_idx = idx_padres[0]
        
        # Criterio estricto: El padre solo entra si mejora al peor hijo
        if apt_pob[best_padre_idx] < nuevas_apt[idx_peor_nueva]:
            nueva_pob[idx_peor_nueva] = poblacion[best_padre_idx][:]
            nuevas_apt[idx_peor_nueva] = apt_pob[best_padre_idx]

    return (nueva_pob, nuevas_apt)
    