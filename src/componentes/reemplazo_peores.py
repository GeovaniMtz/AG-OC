from typing import List, Sequence, Tuple
import math

def reemplazo_peores(
    poblacion: List[List[float]],
    hijos: List[List[float]],
    apt_pob: Sequence[float],
    apt_hijos: Sequence[float],
    porcentaje: float = 1.0,   # Generacion completa default
    elitismo: int = 1          # cuántos mejores padres preservo
) -> Tuple[List[List[float]], List[float]]:
    """
    Minimización: reemplaza los k peores padres por los k mejores hijos.
    Garantiza 'elitismo' mejores padres preservados.
    Devuelve (nueva_poblacion, nuevas_aptitudes).
    """
    N = len(poblacion)
    assert len(apt_pob) == N and len(hijos) == len(apt_hijos), "Longitudes inconsistentes"

    # Clamp de porcentaje y cálculo con floor
    p = max(0.0, min(1.0, porcentaje))
    k = int(math.floor(N * p))
    k = min(k, len(hijos))  # no puedes reemplazar más que los hijos disponibles

    # Si no hay reemplazo, regresa copias
    if k == 0:
        return ([ind[:] for ind in poblacion], list(apt_pob))

    # Índices padres ordenados por aptitud ascendente (mejores primero)
    idx_padres = sorted(range(N), key=lambda i: apt_pob[i])

    # Asegura que 'elitismo' no exceda N
    e = max(0, min(elitismo, N))
    elite_idx = set(idx_padres[:e])

    # === Caso generacional completo ===
    if k == N:
        # Tomo los N mejores hijos
        idx_mej_hijos = sorted(range(len(hijos)), key=lambda i: apt_hijos[i])[:N]
        nueva_pob = [hijos[i][:] for i in idx_mej_hijos]
        nuevas_apt = [apt_hijos[i] for i in idx_mej_hijos]

        # Reinsertar élite adulto si mejora al peor hijo (o si quieres, asegurar al menos 1 élite)
        if e >= 1:
            idx_peor_nueva = max(range(N), key=lambda i: nuevas_apt[i])
            # mejor padre (índice global)
            best_padre = idx_padres[0]
            if apt_pob[best_padre] < nuevas_apt[idx_peor_nueva]:
                nueva_pob[idx_peor_nueva] = poblacion[best_padre][:]
                nuevas_apt[idx_peor_nueva] = apt_pob[best_padre]

        return (nueva_pob, nuevas_apt)

    # Ejercicio 2b: variante de reemplazo parcial
    # === Caso reemplazo parcial ===
    # Candidatos a reemplazo (excluyendo élite): peores primero
    candidatos = [i for i in idx_padres[e:]]          # padres sin élite
    candidatos_peores = list(reversed(candidatos))[:k] # tomar k peores

    # Elegimos los k mejores hijos
    idx_mej_hijos = sorted(range(len(hijos)), key=lambda i: apt_hijos[i])[:k]

    nueva_pob = [ind[:] for ind in poblacion]
    nuevas_apt = list(apt_pob)

    # Mapa rápido pos_adulto -> idx_hijo
    asignacion = dict(zip(candidatos_peores, idx_mej_hijos))

    for pos, h_idx in asignacion.items():
        nueva_pob[pos]  = hijos[h_idx][:]
        nuevas_apt[pos] = apt_hijos[h_idx]

    # Garantía de elitismo: si por accidente tocamos élite, reinsertamos
    # (no debería pasar porque los excluimos, pero queda como cinturón)
    for pos_elite in elite_idx:
        # si el élite no está idéntico (o fue reemplazado por error), lo reponemos
        pass  # no se tocó en este flujo

    return (nueva_pob, nuevas_apt)
