import numpy as np


def kmeans(X, k, iteraciones=100, semilla=0):
    rng = np.random.default_rng(semilla)
    centros = X[rng.choice(len(X), size=k, replace=False)]
    for _ in range(iteraciones):
        d2 = (X ** 2).sum(1)[:, None] + (centros ** 2).sum(1)[None, :] - 2 * X @ centros.T
        grupo = d2.argmin(axis=1)
        nuevos = np.array([X[grupo == j].mean(axis=0) if (grupo == j).any() else centros[j]
                           for j in range(k)])
        if np.allclose(nuevos, centros):
            break
        centros = nuevos
    return grupo, centros
