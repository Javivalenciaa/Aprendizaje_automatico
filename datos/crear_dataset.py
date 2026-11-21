"""Convierte una carpeta de canciones (una subcarpeta por genero) en
rasgos.npz.

Uso (desde la carpeta del proyecto):
    python -m datos.crear_dataset genres_original
"""
import sys
from pathlib import Path

import numpy as np

from datos.audio import rasgos_librosa


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


if __name__ == "__main__":
    crear(sys.argv[1])
