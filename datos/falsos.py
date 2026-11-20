import numpy as np


def datos_falsos(n_por_genero=100, n_rasgos=40, n_generos=10, semilla=1):
    """Inventa canciones: cada género tiene un centro y las canciones se
    reparten alrededor.
    """
    rng = np.random.default_rng(semilla)
    centros = rng.standard_normal((n_generos, n_rasgos)) * 1.2
    X = np.vstack([centros[g] + rng.standard_normal((n_por_genero, n_rasgos)) * 2.0
                   for g in range(n_generos)])
    y = np.repeat(np.arange(n_generos), n_por_genero)
    return X, y