"""Graficas para la presentacion: matriz de confusion y espectrograma.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def dibujar_confusion(M, generos, archivo="confusion.png"):
    plt.figure(figsize=(7, 6))
    plt.imshow(M, cmap="Blues")
    plt.colorbar()
    plt.xticks(range(len(generos)), generos, rotation=45, ha="right")
    plt.yticks(range(len(generos)), generos)
    for i in range(len(generos)):
        for j in range(len(generos)):
            plt.text(j, i, M[i, j], ha="center", va="center",
                     color="white" if M[i, j] > M.max() / 2 else "black")
    plt.xlabel("genero que dijo la red")
    plt.ylabel("genero real")
    plt.tight_layout()
    plt.savefig(archivo, dpi=150)
    plt.close()


def dibujar_espectrograma(ruta, archivo="espectrograma.png"):
    import librosa
    from datos.audio import espectrograma
    senal, sr = librosa.load(ruta, sr=22050, duration=30)
    S = np.log1p(espectrograma(senal)).T
    plt.figure(figsize=(8, 4))
    plt.imshow(S, origin="lower", aspect="auto", cmap="magma",
               extent=[0, len(senal) / sr, 0, sr / 2])
    plt.xlabel("tiempo (s)")
    plt.ylabel("frecuencia (Hz)")
    plt.colorbar(label="intensidad (log)")
    plt.tight_layout()
    plt.savefig(archivo, dpi=150)
    plt.close()


if __name__ == "__main__":
    import sys
    M = np.array([[8, 2, 0], [1, 7, 2], [0, 1, 9]])
    dibujar_confusion(M, ["blues", "jazz", "metal"])
    dibujar_espectrograma(sys.argv[1])
    print("guardadas confusion.png y espectrograma.png")
