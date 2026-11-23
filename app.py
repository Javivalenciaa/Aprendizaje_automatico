"""Web: sube una cancion y la red dice de que genero parece. Se ejecuta con:
streamlit run app.py"""
import tempfile

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from nn.red import crear_mlp, cargar_pesos
from datos.audio import cargar_centro, rasgos_segmentos, espectrograma
from datos.preparar import normalizar

ICONOS = {"blues": "🎷", "classical": "🎻", "country": "🤠", "disco": "🪩",
          "hiphop": "🎤", "jazz": "🎺", "metal": "🤘", "pop": "💿",
          "reggae": "🌴", "rock": "🎸"}

st.set_page_config(page_title="Clasificador de genero musical", page_icon="🎧",
                   layout="centered")

st.markdown("""
<style>
.ganador {padding: 1.4rem 1.6rem; border-radius: 16px; color: white;
          background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
          margin: 0.5rem 0 1rem 0;}
.ganador .etiqueta {font-size: 0.9rem; opacity: 0.85; margin: 0;}
.ganador .genero {font-size: 2.4rem; font-weight: 700; margin: 0.1rem 0;}
.ganador .confianza {font-size: 1.1rem; margin: 0;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def cargar_modelo():
    d = np.load("modelo.npz", allow_pickle=False)
    generos = d["generos"]
    red = crear_mlp(40, d["ocultas"].tolist(), len(generos))
    cargar_pesos(red, "modelo.npz")
    return red, d["mu"], d["sigma"], generos


def dibujar_espectrograma(ruta):
    import librosa
    import matplotlib.pyplot as plt
    senal, sr = cargar_centro(ruta)
    S = np.log1p(espectrograma(senal)).T
    fig, ax = plt.subplots(figsize=(8, 2.6))
    ax.imshow(S, origin="lower", aspect="auto", cmap="magma",
              extent=[0, len(senal) / sr, 0, sr / 2])
    ax.set_xlabel("tiempo (s)")
    ax.set_ylabel("frecuencia (Hz)")
    fig.tight_layout()
    return fig


red, mu, sigma, generos = cargar_modelo()

with st.sidebar:
    st.header("🎧 Cómo funciona")
    st.markdown(
        "1. Se cogen **30 s del centro** de la canción y se cortan en **trozos "
        "de 3 s**. Cada trozo se resume en **40 números** (rasgos MFCC).\n"
        "2. Se **normalizan** con la media y la desviación del entrenamiento.\n"
        "3. Una **red neuronal hecha a mano** (40 → 128 → 64 → 10) da una "
        "probabilidad para cada género a cada trozo.\n"
        "4. Se **promedian** las probabilidades de todos los trozos.")
    st.subheader("Géneros que conoce")
    st.markdown("  \n".join(f"{ICONOS.get(g, '🎵')} {g}" for g in generos))
    st.caption("Solo acepta .wav, analiza 30 segundos del centro y siempre "
               "elige uno de estos 10 géneros.")

st.title("🎶 ¿De qué género es tu canción?")
st.write("Sube un archivo **.wav** y la red te dirá a qué género se parece.")

archivo = st.file_uploader("Sube una cancion (.wav)", type=["wav"])

if archivo is None:
    st.info("👆 Prueba con una canción de la carpeta `genres_original`.")
else:
    st.audio(archivo)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(archivo.getvalue())
        ruta = f.name

    with st.spinner("Escuchando la canción..."):
        senal, _ = cargar_centro(ruta)
        x = normalizar(rasgos_segmentos(senal), mu, sigma)   # un fila por trozo de 3 s
        p = red.predecir_proba(x).mean(axis=0)               # media de todos los trozos
    mejor = int(p.argmax())

    st.markdown(f"""
    <div class="ganador">
      <p class="etiqueta">Parece</p>
      <p class="genero">{ICONOS.get(generos[mejor], '🎵')} {generos[mejor]}</p>
      <p class="confianza">con un {p[mejor]:.0%} de probabilidad</p>
    </div>
    """, unsafe_allow_html=True)

    orden = np.argsort(p)[::-1]
    col1, col2 = st.columns(2)
    for puesto, i in enumerate(orden[1:3], start=2):
        (col1 if puesto == 2 else col2).metric(
            f"{puesto}º  {ICONOS.get(generos[i], '🎵')} {generos[i]}", f"{p[i]:.0%}")

    st.subheader("Probabilidad de cada género")
    tabla = pd.DataFrame({"genero": generos, "probabilidad": p,
                          "ganador": ["sí" if i == mejor else "no"
                                      for i in range(len(generos))]})
    grafico = alt.Chart(tabla).mark_bar(cornerRadiusEnd=6).encode(
        x=alt.X("probabilidad:Q", axis=alt.Axis(format="%"), title=None,
                scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("genero:N", sort="-x", title=None),
        color=alt.Color("ganador:N", legend=None,
                        scale=alt.Scale(domain=["sí", "no"],
                                        range=["#6a11cb", "#b8c4e0"])),
        tooltip=["genero", alt.Tooltip("probabilidad:Q", format=".1%")],
    ).properties(height=320)
    st.altair_chart(grafico, use_container_width=True)

    with st.expander("🔊 Ver el espectrograma de la canción"):
        st.pyplot(dibujar_espectrograma(ruta))
        st.caption("Eje horizontal: tiempo. Eje vertical: frecuencia (abajo "
                   "graves, arriba agudos). El color es la intensidad.")
