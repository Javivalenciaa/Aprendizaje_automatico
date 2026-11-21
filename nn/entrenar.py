import numpy as np
from .perdidas import entropia_cruzada


def evaluar(red, X, y):
    Z = red.forward(X)
    perdida, _ = entropia_cruzada(Z, y)
    acc = (Z.argmax(axis=1) == y).mean()
    return perdida, acc


def entrenar(
    red,
    optimizador,
    Xtr,
    ytr,
    Xval,
    yval,
    epocas=200,
    tam_lote=32,
    l2=0.0,
    paciencia=20,
    semilla=0,
    verbose=False,
):
    rng = np.random.default_rng(semilla)
    N = len(Xtr)
    hist = {
        "perdida_train": [],
        "perdida_val": [],
        "acc_train": [],
        "acc_val": [],
    }
    mejor, mejores_pesos, sin_mejora = np.inf, None, 0

    for ep in range(epocas):
        orden = rng.permutation(N)
        for i in range(0, N, tam_lote):
            idx = orden[i : i + tam_lote]
            Z = red.forward(Xtr[idx])
            _, dZ = entropia_cruzada(Z, ytr[idx])
            red.backward(dZ)
            if l2 > 0:
                for p, g in red.params():
                    if p.ndim == 2:
                        g += l2 * p
            optimizador.paso(red.params())

        lt, at = evaluar(red, Xtr, ytr)
        lv, av = evaluar(red, Xval, yval)
        for k, v in zip(hist, (lt, lv, at, av)):
            hist[k].append(v)

        if verbose and ep % 10 == 0:
            print(f"epoca {ep:3d}  perdida val {lv:.3f}  acc val {av:.3f}")

        if lv < mejor - 1e-4:
            mejor, sin_mejora = lv, 0
            mejores_pesos = [p.copy() for p, _ in red.params()]
        else:
            sin_mejora += 1
            if sin_mejora >= paciencia:
                break

    for (p, _), q in zip(red.params(), mejores_pesos):
        p[...] = q
    return hist