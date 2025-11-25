import csv
from typing import List
from main_ga import ejecutar_ga_real

def correr_experimentos(
    nombre_archivo: str = "resultados_ga.csv",
    funciones: List[str] = None,
    cruzas: List[str] = None,
    dim: int = 10,
    tam_pob: int = 50,
    generaciones: int = 500,
    repeticiones: int = 20,
):
    """
    Corre varias veces el AG para distintas funciones y operadores de cruza,
    y guarda los resultados (incluyendo tiempo) en un CSV.

    Cada fila del CSV corresponde a UNA corrida.
    """
    if funciones is None:
        funciones = ["sphere", "rastrigin"]  # puedes agregar ackley, griewank, rosenbrock

    if cruzas is None:
        cruzas = ["un_punto", "uniforme", "blx", "sbx"]

    with open(nombre_archivo, mode="w", newline="") as f:
        writer = csv.writer(f)

        # Encabezados del CSV
        writer.writerow([
            "funcion",
            "tipo_cruza",
            "dim",
            "tam_pob",
            "generaciones",
            "repeticion",
            "semilla",
            "mejor_final",
            "peor_final",
            "promedio_final",
            "tiempo_total_seg",
        ])

        rep_global = 0

        for nombre_func in funciones:
            for tipo_cruza in cruzas:
                for rep in range(repeticiones):
                    # Semilla distinta por corrida para poder promediar
                    semilla = 1000 * rep_global + 123
                    rep_global += 1

                    print(f"[INFO] Función={nombre_func}, cruza={tipo_cruza}, rep={rep+1}/{repeticiones}, semilla={semilla}")

                    resultado = ejecutar_ga_real(
                        nombre_func=nombre_func,
                        dim=dim,
                        tam_pob=tam_pob,
                        generaciones=generaciones,
                        pc=0.9,
                        tipo_cruza=tipo_cruza,
                        porcentaje_reemplazo=1.0,
                        elitismo=1,
                        semilla=semilla,
                        alpha_blx=0.5,
                        eta_c_sbx=10.0,
                        amplitud_mut=0.1,
                    )

                    writer.writerow([
                        resultado["nombre_func"],
                        resultado["tipo_cruza"],
                        resultado["dim"],
                        resultado["tam_pob"],
                        resultado["generaciones"],
                        rep,
                        resultado["semilla"],
                        resultado["mejor_final"],
                        resultado["peor_final"],
                        resultado["promedio_final"],
                        resultado["tiempo_total"],
                    ])

    print(f"\n[OK] Resultados guardados en: {nombre_archivo}")


if __name__ == "__main__":
    # Ajusta estos parámetros como quieras para tus experimentos
    correr_experimentos(
        nombre_archivo="resultados_ga_sphere_rastrigin.csv",
        funciones=["sphere", "rastrigin", "rosenbrock"], # se pueden agregar más
        cruzas=["un_punto", "uniforme", "blx", "sbx"],
        dim=10,
        tam_pob=50,
        generaciones=500,
        repeticiones=20,
    )
