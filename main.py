"""Entrena la red y guarda modelo.npz (lo que carga la web).

Antes hay que tener rasgos.npz (python -m datos.crear_dataset genres_original).
Uso:  python main.py
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nn.red import crear_mlp, guardar
from nn.optimizadores import Adam
from nn.entrenar import entrenar, evaluar
from datos.preparar import dividir, ajustar_normalizacion, normalizar
from datos.metricas import accuracy, matriz_confusion, precision_recall_f1

OCULTAS = [64, 32]
LR = 1e-3
L2 = 1e-4
EPOCAS = 300
PACIENCIA = 20

d = np.load("rasgos.npz", allow_pickle=False)
X, y, generos = d["X"], d["y"], d["generos"]
C = len(generos)
print(f"{len(y)} canciones, {X.shape[1]} rasgos, {C} generos")

(Xtr, ytr), (Xva, yva), (Xte, yte) = dividir(X, y)
mu, sigma = ajustar_normalizacion(Xtr)
Xtr, Xva, Xte = (normalizar(A, mu, sigma) for A in (Xtr, Xva, Xte))
print(f"entrenamiento {len(ytr)} · validacion {len(yva)} · test {len(yte)}")


def entrenar_modelo(ocultas):
    red = crear_mlp(X.shape[1], ocultas, C)
    hist = entrenar(red, Adam(lr=LR), Xtr, ytr, Xva, yva,
                    epocas=EPOCAS, l2=L2, paciencia=PACIENCIA, verbose=True)
    return red, hist


print("\n== Red", OCULTAS, "==")
red, hist = entrenar_modelo(OCULTAS)
print("\n== Modelo lineal (sin capas ocultas) ==")
lineal, hist_lin = entrenar_modelo([])

# El test se mira UNA sola vez, al final.
print("\n== Resultados en test ==")
for nombre, m in (("red", red), ("lineal", lineal)):
    _, acc = evaluar(m, Xte, yte)
    print(f"{nombre:7s} accuracy test: {acc:.3f}   (azar: {1 / C:.3f})")

M = matriz_confusion(yte, red.predecir(Xte), C)
p, r, f1 = precision_recall_f1(M)
print("\ngenero      precision  recall    F1")
for g, a, b, c in zip(generos, p, r, f1):
    print(f"{g:10s}  {a:8.2f} {b:7.2f} {c:7.2f}")
print("\nmatriz de confusion (filas = real, columnas = predicho):")
print(M)

guardar(red, "modelo.npz", mu=mu, sigma=sigma, generos=generos,
        ocultas=np.array(OCULTAS))
print("\nGuardado modelo.npz")

plt.figure(figsize=(7, 4))
plt.plot(hist["perdida_train"], label="entrenamiento")
plt.plot(hist["perdida_val"], label="validacion")
plt.xlabel("epoca")
plt.ylabel("perdida")
plt.legend()
plt.tight_layout()
plt.savefig("curvas.png", dpi=130)
print("Guardado curvas.png")
