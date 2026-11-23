import numpy as np


def rasgos_librosa(ruta, sr=22050, duracion=30, n_mfcc=20):
    """Version rapida: librosa calcula los MFCC. Devuelve un vector de
    2*n_mfcc = 40 numeros.
    """
    import librosa
    senal, sr = librosa.load(ruta, sr=sr, duration=duracion)
    mfcc = librosa.feature.mfcc(y=senal, sr=sr, n_mfcc=n_mfcc)
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])


def cargar_centro(ruta, sr=22050, duracion=30):
    """Carga 30 s del CENTRO de la cancion (el principio suele ser una
    intro que no representa al genero)."""
    import librosa
    total = librosa.get_duration(path=ruta)
    inicio = max(0, total / 2 - duracion / 2)
    return librosa.load(ruta, sr=sr, offset=inicio, duration=duracion)


def rasgos_segmentos(senal, sr=22050, seg=3):
    """Corta la cancion en trozos de 'seg' segundos y saca 43 rasgos por trozo.

    Timbre (40): se iguala el volumen del trozo, se calculan 21 MFCC, se tira
    el primero (solo mide el volumen) y se guardan la media y la desviacion
    de los 20 restantes.
    Ritmo (3): cuanta "actividad ritmica" hay en el trozo (media y desviacion
    de la fuerza de los golpes) y el tempo de la cancion en pulsos por minuto.
    Devuelve una matriz (numero de trozos x 43).
    """
    import librosa

    def igualar(t):
        return t / (np.sqrt(np.mean(t ** 2)) + 1e-9)

    golpes = librosa.onset.onset_strength(y=igualar(senal), sr=sr)
    calcular_tempo = getattr(librosa.feature, "tempo", None) or librosa.beat.tempo
    bpm = float(calcular_tempo(onset_envelope=golpes, sr=sr)[0])
    n = int(seg * sr)
    filas = []
    for i in range(len(senal) // n):
        trozo = igualar(senal[i * n:(i + 1) * n])
        mfcc = librosa.feature.mfcc(y=trozo, sr=sr, n_mfcc=21)[1:]
        o = librosa.onset.onset_strength(y=trozo, sr=sr)
        filas.append(np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1),
                                     [o.mean(), o.std(), bpm]]))
    return np.array(filas)


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
