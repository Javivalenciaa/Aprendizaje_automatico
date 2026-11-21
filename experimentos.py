
import os
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nn.red import crear_mlp
from nn.optimizadores import SGD, Momentum, Adam
from nn.entrenar import entrenar
from datos.preparar import dividir, ajustar_normalizacion, normalizar

if os.path.exists("rasgos.npz"):
    d = np.load("rasgos.npz", allow_pickle=False)
    X, y = d["X"], d["y"]
else:
    from datos.falsos import datos_falsos

    X, y = datos_falsos()

C = int(y.max()) + 1
(Xtr, ytr), (Xva, yva), _ = dividir(X, y)
mu, sigma = ajustar_normalizacion(Xtr)
Xtr, Xva = normalizar(Xtr, mu, sigma), normalizar(Xva, mu, sigma)


def correr(ocultas, optimizador, **kw):
    red = crear_mlp(X.shape[1], ocultas, C)
    return entrenar(red, optimizador, Xtr, ytr, Xva, yva, epocas=200, **kw)


def dibujar(curvas, titulo, archivo, clave="perdida_val"):
    plt.figure(figsize=(7, 4))
    for nombre, hist in curvas.items():
        plt.plot(hist[clave], label=nombre)
    plt.xlabel("epoca")
    plt.ylabel(clave)
    plt.title(titulo)
    plt.legend()
    plt.tight_layout()
    plt.savefig(archivo, dpi=150)
    plt.close()
    print("guardada", archivo)


# 1. Optimizadores
dibujar(
    {
        "SGD": correr([64, 32], SGD(0.1)),
        "Momentum": correr([64, 32], Momentum(0.05, 0.9)),
        "Adam": correr([64, 32], Adam(1e-3)),
    },
    "Perdida de validacion por optimizador",
    "exp1_optimizadores.png",
)

# 2. Tasa de aprendizaje
dibujar(
    {
        f"lr={lr}": correr([64, 32], Adam(lr))
        for lr in (1e-1, 1e-2, 1e-3, 1e-4)
    },
    "Efecto de la tasa de aprendizaje (Adam)",
    "exp2_tasa_aprendizaje.png",
)

# 3. Regularizacion L2: se ve en la separacion entre train (rayas) y validacion (linea)
plt.figure(figsize=(7, 4))
for l2 in (0.0, 1e-3, 1e-1):
    h = correr([64, 32], Adam(1e-3), l2=l2)
    plt.plot(h["acc_train"], "--", label=f"L2={l2} train")
    plt.plot(h["acc_val"], "-", label=f"L2={l2} val")
plt.xlabel("epoca")
plt.ylabel("acierto")
plt.title("Efecto de L2")
plt.legend()
plt.tight_layout()
plt.savefig("exp3_l2.png", dpi=150)
plt.close()
print("guardada exp3_l2.png")

# 4. Tamano de la red (con [] = modelo lineal, la referencia)
dibujar(
    {
        "sin ocultas (lineal)": correr([], Adam(1e-3)),
        "[16]": correr([16], Adam(1e-3)),
        "[64, 32]": correr([64, 32], Adam(1e-3)),
        "[128, 64]": correr([128, 64], Adam(1e-3)),
    },
    "Acierto de validacion segun el tamano de la red",
    "exp4_tamano.png",
    clave="acc_val",
)