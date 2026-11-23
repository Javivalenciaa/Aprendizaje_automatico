"""Entrena la red y guarda modelo.npz (lo que carga la web).

Antes hay que tener rasgos_seg.npz (ya viene en el repo; para rehacerlo:
python -m datos.crear_dataset genres_original trozos).
Uso:  python main.py
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nn.red import crear_mlp, guardar
from nn.optimizadores import Adam
from nn.entrenar import entrenar
from datos.preparar import cargar_segmentos, ajustar_normalizacion, normalizar
from datos.metricas import accuracy, matriz_confusion, precision_recall_f1, votar_por_cancion

OCULTAS = [128, 64]
LR = 1e-3
L2 = 1e-3
EPOCAS = 300
PACIENCIA = 15

d = cargar_segmentos()
generos = d["generos"]
C = len(generos)
mu, sigma = ajustar_normalizacion(d["Xtr"])
Xtr, Xva, Xte = (normalizar(d[k], mu, sigma) for k in ("Xtr", "Xva", "Xte"))
ytr, yva, yte, cte = d["ytr"], d["yva"], d["yte"], d["cte"]
print(f"trozos de 3 s -> entrenamiento {len(ytr)} · validacion {len(yva)} · test {len(yte)}")


def entrenar_modelo(ocultas):
    red = crear_mlp(Xtr.shape[1], ocultas, C)
    hist = entrenar(red, Adam(lr=LR), Xtr, ytr, Xva, yva,
                    epocas=EPOCAS, l2=L2, paciencia=PACIENCIA, verbose=True)
    return red, hist


print("\n== Red", OCULTAS, "==")
red, hist = entrenar_modelo(OCULTAS)
print("\n== Modelo lineal (sin capas ocultas) ==")
lineal, hist_lin = entrenar_modelo([])

# El test se mira UNA sola vez, al final. La nota es por CANCION: se promedian
# las probabilidades de todos sus trozos.
print("\n== Resultados en test (por cancion) ==")
for nombre, m in (("red", red), ("lineal", lineal)):
    real, pred = votar_por_cancion(m.predecir_proba(Xte), cte, yte)
    print(f"{nombre:7s} accuracy test: {accuracy(real, pred):.3f}   (azar: {1 / C:.3f})")

real, pred = votar_por_cancion(red.predecir_proba(Xte), cte, yte)
M = matriz_confusion(real, pred, C)
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
