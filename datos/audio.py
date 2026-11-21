import numpy as np


def rasgos_librosa(ruta, sr=22050, duracion=30, n_mfcc=20):
    """Version rapida: librosa calcula los MFCC. Devuelve un vector de
    2*n_mfcc = 40 numeros.
    """
    import librosa
    senal, sr = librosa.load(ruta, sr=sr, duration=duracion)
    mfcc = librosa.feature.mfcc(y=senal, sr=sr, n_mfcc=n_mfcc)
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])


def espectrograma(senal, n_fft=1024, salto=512):
    """Espectrograma a mano: ventanas que se solapan + transformada de
    Fourier.
    """
    ventana = np.hanning(n_fft)
    n_marcos = 1 + (len(senal) - n_fft) // salto
    marcos = np.stack([senal[i * salto:i * salto + n_fft] * ventana
                       for i in range(n_marcos)])
    return np.abs(np.fft.rfft(marcos, axis=1)) ** 2


def rasgos_a_mano(senal, n_bandas=20):
    """Plan B sin librosa: log-energia en n_bandas de frecuencia; media y
    desviacion en el tiempo.
    """
    S = espectrograma(senal)
    bandas = np.array_split(np.arange(S.shape[1]), n_bandas)
    E = np.log1p(np.stack([S[:, b].mean(axis=1) for b in bandas], axis=1))
    return np.concatenate([E.mean(axis=0), E.std(axis=0)])
