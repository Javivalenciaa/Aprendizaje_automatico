"""Distorsiones para ensenar a la red que una cancion sigue siendo del mismo
genero aunque suene distinta (otra grabacion, otro master, otro volumen)."""
import numpy as np
from scipy.signal import butter, sosfilt


def distorsionar(senal, rng, sr=22050):
    y = senal
    if rng.random() < 0.6:   # quitar agudos
        sos = butter(4, rng.uniform(2500, 10000), "low", fs=sr, output="sos")
        y = sosfilt(sos, y)
    if rng.random() < 0.6:   # quitar graves
        sos = butter(4, rng.uniform(60, 500), "high", fs=sr, output="sos")
        y = sosfilt(sos, y)
    if rng.random() < 0.5:   # saturar (sonido "fuerte" de master moderno)
        y = np.tanh(rng.uniform(1, 6) * y / (np.abs(y).max() + 1e-9))
    return y.astype(np.float32)
