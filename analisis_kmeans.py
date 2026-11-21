import numpy as np

from datos.preparar import dividir, ajustar_normalizacion, normalizar
from datos.kmeans import kmeans

d = np.load("rasgos.npz", allow_pickle=False)
X, y, generos = d["X"], d["y"], d["generos"]

(Xtr, ytr), _, _ = dividir(X, y)
mu, sigma = ajustar_normalizacion(Xtr)
Xtr = normalizar(Xtr, mu, sigma)

k = 10
grupo, centros = kmeans(Xtr, k)

T = np.zeros((k, len(generos)), dtype=int)
for g, genero in zip(grupo, ytr):
    T[g, genero] += 1

print("pureza:", round(T.max(axis=1).sum() / T.sum(), 3))
for j in range(k):
    mejores = np.argsort(T[j])[::-1][:3]
    print(f"grupo {j}: " + ", ".join(f"{generos[c]} {T[j, c]}" for c in mejores))
