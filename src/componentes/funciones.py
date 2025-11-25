import numpy as np

# A. Función Sphere
def sphere(x):
    """
    Función Esfera.
    Mínimo global: f(0, 0, ..., 0) = 0.
    """
    x = np.array(x)  # Asegura que siempre sea un arreglo de numpy
    return np.sum(x**2)

# B. Función Ackley
def ackley(x):
    """
    Función de Ackley.
    Mínimo global: f(0, 0, ..., 0) = 0.
    """
    x = np.array(x)  # asegura que siempre sea arreglo numpy
    n = x.size
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(2 * np.pi * x))
    term1 = -20 * np.exp(-0.2 * np.sqrt(sum_sq / n))
    term2 = -np.exp(sum_cos / n)
    return 20 + np.e + term1 + term2

# C. Función Griewank
def griewank(x):
    """
    Función de Griewank.
    Mínimo global: f(0, 0, ..., 0) = 0.
    """
    x = np.array(x)
    n = x.size
    # Índices para el producto (de 1 a n)
    i = np.arange(1, n + 1)
    sum_term = np.sum(x**2 / 4000)
    prod_term = np.prod(np.cos(x / np.sqrt(i)))
    return 1 + sum_term - prod_term

# D. Función Rastrigin
def rastrigin(x):
    """
    Función de Rastrigin. Altamente multimodal.
    Mínimo global: f(0, 0, ..., 0) = 0.
    """
    x = np.array(x)
    n = x.size
    sum_term = np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
    return 10 * n + sum_term

# E. Función Rosenbrock
def rosenbrock(x):
    """
    Función de Rosenbrock (Banana).
    Mínimo global: f(1, 1, ..., 1) = 0.
    """
    x = np.array(x)  # asegura que siempre sea arreglo numpy
    n = x.size
    if n < 2:
        raise ValueError("La función Rosenbrock requiere al menos 2 dimensiones")
    suma = 0
    for i in range(n - 1):
        suma += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
    return suma

if __name__ == "__main__":
    # Pruebas de la función Sphere
    print("--- Función Sphere ---")
    print("Sphere ([0, 0]):", sphere([0, 0]))
    print("Sphere ([1, 2, 3]):", sphere([1, 2, 3]))
    print("Sphere ([0,0,...,0]):", sphere(np.zeros(10)))
    print("-" * 25)
    
    # Pruebas de la función Ackley
    print("\n--- Función Ackley ---")
    print("Ackley ([0,0]):", ackley([0, 0]))
    print("Ackley ([1,2,3]):", ackley([1, 2, 3]))
    print("Ackley ([0,0,0,0,0,0,0,0,0,0]):", ackley(np.zeros(10)))
    print("-" * 25)
    
    # Pruebas de la función Griewank
    print("\n--- Función Griewank ---")
    print("Griewank ([0, 0]):", griewank([0, 0]))
    print("Griewank ([1, 2, 3]):", griewank([1, 2, 3]))
    print("Griewank ([0,0,...,0]):", griewank(np.zeros(10)))
    print("-" * 25)
    
    # Pruebas de la función Rastrigin
    print("\n--- Función Rastrigin ---")
    print("Rastrigin ([0, 0]):", rastrigin([0, 0]))
    print("Rastrigin ([1, 2, 3]):", rastrigin([1, 2, 3]))
    print("Rastrigin ([4.5, 4.5]):", rastrigin([4.5, 4.5]))
    print("Rastrigin ([0,0,...,0]):", rastrigin(np.zeros(10)))
    print("-" * 25)
    
    # Pruebas de la función Rosenbrock
    print("\n--- Función Rosenbrock ---")
    print("Rosenbrock([1,1]):", rosenbrock([1, 1]))
    print("Rosenbrock([0,0]):", rosenbrock([0, 0]))
    print("Rosenbrock([1,1,1]):", rosenbrock([1, 1, 1]))
    print("Rosenbrock([0,1,2]):", rosenbrock([0, 1, 2]))
    print("Rosenbrock(np.ones(10)):", rosenbrock(np.ones(10)))