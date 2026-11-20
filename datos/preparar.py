import numpy as np


def dividir(X, y, prop_val=0.15, prop_test=0.15, semilla=0):
    rng = np.random.default_rng(semilla)
    tr, va, te = [], [], []
    for clase in np.unique(y):
        idx = rng.permutation(np.where(y == clase)[0])
        n_te = int(round(len(idx) * prop_test))
        n_va = int(round(len(idx) * prop_val))
        te += list(idx[:n_te])
        va += list(idx[n_te:n_te + n_va])
        tr += list(idx[n_te + n_va:])
    tr, va, te = map(np.array, (tr, va, te))
    return (X[tr], y[tr]), (X[va], y[va]), (X[te], y[te])


def ajustar_normalizacion(Xtr):
    return Xtr.mean(axis=0), Xtr.std(axis=0)


def normalizar(X, mu, sigma):
    return (X - mu) / (sigma + 1e-8)
