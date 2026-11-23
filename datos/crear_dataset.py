"""Convierte una carpeta de canciones (una subcarpeta por genero) en
rasgos.npz.

Uso (desde la carpeta del proyecto):
    python -m datos.crear_dataset genres_original
"""
import sys
from pathlib import Path

import numpy as np

from datos.audio import rasgos_librosa, rasgos_segmentos
from datos.aumentar import distorsionar


def crear(carpeta, salida="rasgos.npz"):
    carpeta = Path(carpeta)
    generos = sorted(p.name for p in carpeta.iterdir() if p.is_dir())
    X, y = [], []
    for i, genero in enumerate(generos):
        for archivo in sorted((carpeta / genero).glob("*.wav")):
            try:
                X.append(rasgos_librosa(str(archivo)))
                y.append(i)
            except Exception as e:
                print("saltado:", archivo.name, "-", e)
    np.savez(salida, X=np.array(X), y=np.array(y), generos=np.array(generos))
    print(f"{len(y)} canciones, {len(generos)} generos -> {salida}")


def crear_segmentos(carpeta, salida="rasgos_seg.npz", copias=3):
    """Cada cancion -> trozos de 3 s (40 rasgos cada uno). Ademas de la
    original se guardan 'copias' versiones distorsionadas (aug = 1, 2, 3)."""
    import librosa
    carpeta = Path(carpeta)
    generos = sorted(p.name for p in carpeta.iterdir() if p.is_dir())
    X, y, cancion, aug = [], [], [], []
    n = 0
    for i, genero in enumerate(generos):
        for archivo in sorted((carpeta / genero).glob("*.wav")):
            try:
                senal, _ = librosa.load(str(archivo), sr=22050, duration=30)
                versiones = [senal] + [
                    distorsionar(senal, np.random.default_rng(1000 * n + k))
                    for k in range(copias)]
                for k, v in enumerate(versiones):
                    f = rasgos_segmentos(v)
                    X.append(f)
                    y += [i] * len(f)
                    cancion += [n] * len(f)
                    aug += [k] * len(f)
                n += 1
            except Exception as e:
                print("saltado:", archivo.name, "-", e)
    np.savez_compressed(salida, X=np.vstack(X).astype(np.float32),
                        y=np.array(y), cancion=np.array(cancion),
                        aug=np.array(aug), generos=np.array(generos))
    print(f"{n} canciones, {len(y)} trozos -> {salida}")


if __name__ == "__main__":
    # python -m datos.crear_dataset genres_original        (rasgos.npz, version simple)
    # python -m datos.crear_dataset genres_original trozos  (rasgos_seg.npz, el que usa main.py)
    if len(sys.argv) > 2 and sys.argv[2] == "trozos":
        crear_segmentos(sys.argv[1])
    else:
        crear(sys.argv[1])
