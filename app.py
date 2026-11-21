"""Web: sube una cancion y la red dice de que genero parece. Se ejecuta con:
streamlit run app.py"""
import tempfile

import numpy as np
import pandas as pd
import streamlit as st

from nn.red import crear_mlp, cargar_pesos
from datos.audio import rasgos_librosa
from datos.preparar import normalizar

st.title("De que genero es tu cancion?")


@st.cache_resource
def cargar_modelo():
    d = np.load("modelo.npz", allow_pickle=False)
    generos = d["generos"]
    red = crear_mlp(40, d["ocultas"].tolist(), len(generos))
    cargar_pesos(red, "modelo.npz")
    return red, d["mu"], d["sigma"], generos


red, mu, sigma, generos = cargar_modelo()

archivo = st.file_uploader("Sube una cancion (.wav)", type=["wav"])
if archivo is not None:
    st.audio(archivo)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(archivo.getvalue())
        ruta = f.name
    x = normalizar(rasgos_librosa(ruta)[None, :], mu, sigma)
    p = red.predecir_proba(x)[0]
    mejor = int(p.argmax())
    st.subheader(f"Parece {generos[mejor]} ({p[mejor]:.0%})")
    st.bar_chart(pd.Series(p, index=generos))
