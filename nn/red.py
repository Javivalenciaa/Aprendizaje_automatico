import numpy as np
from .capas import Lineal, ReLU
from .perdidas import softmax


class Red:
    """Una lista de capas que se aplican una detras de otra."""

    def __init__(self, capas):
        self.capas = capas

    def forward(self, X):
        for capa in self.capas:
            X = capa.forward(X)
        return X

    def backward(self, dZ):
        for capa in reversed(self.capas):
            dZ = capa.backward(dZ)
        return dZ

    def params(self):
        return [par for capa in self.capas for par in capa.params()]

    def predecir_proba(self, X):
        return softmax(self.forward(X))

    def predecir(self, X):
        return self.forward(X).argmax(axis=1)


def crear_mlp(n_in, ocultas, n_out, semilla=0):
    """crear_mlp(40, [64, 32], 10) -> 40 -> 64 -> 32 -> 10, con ReLU entre capas."""
    rng = np.random.default_rng(semilla)
    capas, previo = [], n_in
    for h in ocultas:
        capas += [Lineal(previo, h, rng), ReLU()]
        previo = h
    capas.append(Lineal(previo, n_out, rng))
    return Red(capas)

def guardar(red, ruta, **extra):
    """Guarda pesos y datos extra (media, desviacion, nombres de generos) en un .npz."""
    datos = {f"p{i}": p for i, (p, _) in enumerate(red.params())}
    datos.update(extra)
    np.savez(ruta, **datos)


def cargar_pesos(red, ruta):
    d = np.load(ruta, allow_pickle=False)
    for i, (p, _) in enumerate(red.params()):
        p[...] = d[f"p{i}"]
    return d
