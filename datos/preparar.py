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


def cargar_segmentos(ruta="rasgos_seg.npz"):
    """Lee rasgos_seg.npz y reparte POR CANCION (todos los trozos de una
    cancion van al mismo grupo: si no, la red veria en entrenamiento trozos
    de las canciones del test).
    - entrenamiento: trozos originales y distorsionados
    - validacion y test: solo los trozos originales
    """
    d = np.load(ruta, allow_pickle=False)
    X, y, cancion, aug = d["X"], d["y"], d["cancion"], d["aug"]
    y_cancion = np.zeros(cancion.max() + 1, dtype=int)
    y_cancion[cancion] = y
    (tr, _), (va, _), (te, _) = dividir(np.arange(len(y_cancion)), y_cancion)
    m_tr = np.isin(cancion, tr)
    m_va = np.isin(cancion, va) & (aug == 0)
    m_te = np.isin(cancion, te) & (aug == 0)
    return dict(Xtr=X[m_tr], ytr=y[m_tr], Xva=X[m_va], yva=y[m_va],
                Xte=X[m_te], yte=y[m_te], cte=cancion[m_te],
                generos=d["generos"])
