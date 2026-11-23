"""Demo para la presentacion: recorre el modelo paso a paso, desde que se sube
la cancion hasta que la red predice el genero. Se ejecuta con:
    streamlit run app_demo.py
(app.py sigue siendo la web sencilla.)"""
import os
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nn.red import crear_mlp, cargar_pesos
from nn.optimizadores import Adam
from nn.entrenar import entrenar
from nn.perdidas import softmax
from datos.audio import espectrograma
from datos.preparar import dividir, normalizar
from datos.metricas import matriz_confusion, precision_recall_f1

POS, NEG, MORADO, GRIS = "#1D9E75", "#D85A30", "#6a11cb", "#b8c4e0"
ICONOS = {"blues": "🎷", "classical": "🎻", "country": "🤠", "disco": "🪩",
          "hiphop": "🎤", "jazz": "🎺", "metal": "🤘", "pop": "💿",
          "reggae": "🌴", "rock": "🎸"}

st.set_page_config(page_title="Del audio a la predicción", page_icon="🎧",
                   layout="wide")
st.markdown("""
<style>
.ganador {padding: 1.2rem 1.6rem; border-radius: 16px; color: white;
          background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);}
.ganador .etiqueta {font-size: 0.9rem; opacity: 0.85; margin: 0;}
.ganador .genero {font-size: 2.4rem; font-weight: 700; margin: 0.1rem 0;}
.ganador .confianza {font-size: 1.1rem; margin: 0;}
.paso {padding: .8rem 1rem; border-left: 5px solid #6a11cb; background: #f4f0fc;
       border-radius: 6px; margin-bottom: 1rem; color: #222;}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ modelo
@st.cache_resource
def cargar_modelo():
    d = np.load("modelo.npz", allow_pickle=False)
    generos = d["generos"]
    red = crear_mlp(40, d["ocultas"].tolist(), len(generos))
    cargar_pesos(red, "modelo.npz")
    return red, d["mu"], d["sigma"], generos, d["ocultas"].tolist()


red, mu, sigma, generos, ocultas = cargar_modelo()
C = len(generos)
n_pesos = sum(p.size for p, _ in red.params())


def es_lineal(capa):
    return hasattr(capa, "W")


# ---------------------------------------------------------------- analisis
def analizar(contenido):
    """Pasa la cancion por todo el recorrido y guarda cada resultado."""
    import librosa
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(contenido)
        ruta = f.name
    senal, sr = librosa.load(ruta, sr=22050, duration=30)
    os.remove(ruta)
    mfcc = librosa.feature.mfcc(y=senal, sr=sr, n_mfcc=20)
    rasgos = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])
    x = normalizar(rasgos[None, :], mu, sigma)

    etapas, h = [], x
    n_lin = 0
    for capa in red.capas:
        h = capa.forward(h)
        if es_lineal(capa):
            n_lin += 1
            etapas.append(("lineal", n_lin, h[0].copy(), capa))
        else:
            etapas.append(("relu", n_lin, h[0].copy(), capa))
    Z = h[0]
    P = softmax(h)[0]
    return dict(senal=senal, sr=sr, mfcc=mfcc, rasgos=rasgos, x=x[0],
                etapas=etapas, Z=Z, P=P)


def barras_con_signo(valores, titulo, etiquetas=None, alto=2.4):
    fig, ax = plt.subplots(figsize=(8, alto))
    colores = [POS if v > 0 else NEG for v in valores]
    ax.bar(range(len(valores)), valores, color=colores)
    ax.axhline(0, color="#444", lw=.8)
    ax.set_title(titulo, fontsize=10)
    if etiquetas is not None:
        ax.set_xticks(range(len(valores)))
        ax.set_xticklabels(etiquetas, rotation=40, ha="right", fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def paso(texto):
    st.markdown(f'<div class="paso">{texto}</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ cabecera
with st.sidebar:
    st.header("🎧 El modelo")
    arq = " → ".join(["40"] + [str(o) for o in ocultas] + [str(C)])
    st.markdown(f"**Red:** {arq}  \n**Pesos y sesgos:** {n_pesos:,}".replace(",", "."))
    st.markdown("**Géneros:**  \n" +
                "  \n".join(f"{ICONOS.get(g, '🎵')} {g}" for g in generos))
    st.caption("Solo acepta .wav. Analiza como mucho los 30 primeros segundos "
               "y siempre elige uno de estos géneros.")

st.title("🎶 Del audio a la predicción")
st.write("Sube una canción y sigue, paso a paso, lo que hace el modelo "
         "hasta decidir su género.")

archivo = st.file_uploader("Sube una canción (.wav)", type=["wav"])
datos = None
if archivo is not None:
    with st.spinner("Analizando la canción..."):
        datos = analizar(archivo.getvalue())

t1, t2, t3, t4, t5, t6 = st.tabs([
    "1 · La canción", "2 · Rasgos (MFCC)", "3 · Normalizar",
    "4 · La red, capa a capa", "5 · Softmax y resultado",
    "6 · ¿Cómo de bueno es el modelo?"])

SIN_CANCION = "👆 Sube una canción para ver este paso."

# ------------------------------------------------------------------- tab 1
with t1:
    if datos is None:
        st.info(SIN_CANCION)
        st.markdown("""
**El recorrido completo**

`audio` → `40 rasgos` → `normalizar` → `capa 1 + ReLU` → `capa 2 + ReLU`
→ `capa de salida` → `softmax` → `género`
""")
    else:
        st.audio(archivo)
        senal, sr = datos["senal"], datos["sr"]
        paso(f"<b>Qué ves:</b> un archivo de audio es una lista enorme de números. "
             f"Esta canción son <b>{len(senal):,}</b> números ({len(senal) / sr:.0f} s a "
             f"{sr} números por segundo). Demasiados y casi todos inútiles para "
             f"distinguir géneros.".replace(",", "."))
        fig, ax = plt.subplots(figsize=(10, 2.2))
        ax.plot(np.arange(len(senal)) / sr, senal, color=MORADO, lw=.4)
        ax.set_xlabel("tiempo (s)")
        ax.set_ylabel("amplitud")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        st.markdown("**Espectrograma:** se corta la canción en trocitos muy cortos "
                    "y se mira cuánta energía hay en graves y en agudos en cada uno.")
        S = np.log1p(espectrograma(senal)).T
        fig, ax = plt.subplots(figsize=(10, 2.8))
        ax.imshow(S, origin="lower", aspect="auto", cmap="magma",
                  extent=[0, len(senal) / sr, 0, sr / 2])
        ax.set_xlabel("tiempo (s)")
        ax.set_ylabel("frecuencia (Hz)")
        fig.tight_layout()
        st.pyplot(fig)
        st.caption("Abajo, graves. Arriba, agudos. El color es la intensidad.")

# ------------------------------------------------------------------- tab 2
with t2:
    if datos is None:
        st.info(SIN_CANCION)
    else:
        paso("<b>Qué ves:</b> de cada trocito de sonido se sacan <b>20 números "
             "(MFCC)</b> que describen la forma de su espectro, de lo general a lo "
             "fino. Luego, para cada MFCC, se calcula su <b>media</b> y su "
             "<b>desviación</b> a lo largo de toda la canción: "
             "20 + 20 = <b>40 rasgos</b>. Esa es la entrada de la red.")
        fig, ax = plt.subplots(figsize=(10, 3))
        im = ax.imshow(datos["mfcc"], origin="lower", aspect="auto", cmap="RdBu_r")
        ax.set_xlabel("trocito de sonido (tiempo →)")
        ax.set_ylabel("MFCC nº")
        fig.colorbar(im, ax=ax, fraction=.03)
        fig.tight_layout()
        st.pyplot(fig)
        st.caption("Cada columna es un trocito de la canción y cada fila un MFCC.")
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(barras_con_signo(datos["rasgos"][:20],
                                       "20 rasgos: la MEDIA de cada MFCC"))
        with c2:
            st.pyplot(barras_con_signo(datos["rasgos"][20:],
                                       "20 rasgos: la DESVIACIÓN de cada MFCC"))
        st.success("La canción de 30 s ha pasado de cientos de miles de números a **40**.")

# ------------------------------------------------------------------- tab 3
with t3:
    if datos is None:
        st.info(SIN_CANCION)
    else:
        paso("<b>Qué ves:</b> los 40 rasgos tienen escalas muy distintas (unos valen "
             "cientos, otros décimas). Se normalizan con la <b>media y la desviación "
             "de las canciones de entrenamiento</b> (guardadas en <code>modelo.npz</code>): "
             "<code>(x − media) / desviación</code>. Siempre las del entrenamiento, "
             "nunca unas nuevas.")
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(barras_con_signo(datos["rasgos"], "Antes: rasgos tal cual", alto=2.8))
        with c2:
            st.pyplot(barras_con_signo(datos["x"], "Después: rasgos normalizados", alto=2.8))
        nombres = [f"media MFCC {i + 1}" for i in range(20)] + \
                  [f"desv. MFCC {i + 1}" for i in range(20)]
        tabla = pd.DataFrame({"rasgo": nombres, "valor": datos["rasgos"],
                              "media del entrenamiento": mu,
                              "desviación del entrenamiento": sigma,
                              "normalizado": datos["x"]}).round(2)
        with st.expander("Ver la cuenta de los 40 rasgos"):
            st.dataframe(tabla, hide_index=True)

# ------------------------------------------------------------------- tab 4
with t4:
    if datos is None:
        st.info(SIN_CANCION)
    else:
        paso("<b>Qué ves:</b> la red aplica capa tras capa. Cada <b>capa lineal</b> "
             "calcula, para cada neurona, <code>entrada · pesos + sesgo</code>. Cada "
             "<b>ReLU</b> deja pasar los positivos y pone a 0 los negativos "
             "(neurona apagada). Verde = positivo, naranja = negativo.")
        entrada = 40
        for tipo, n, valores, capa in datos["etapas"]:
            if tipo == "lineal":
                salida = len(valores)
                es_salida = capa is [c for c in red.capas if es_lineal(c)][-1]
                nombre = "Capa de salida (una puntuación por género)" if es_salida \
                    else f"Capa oculta {n}"
                st.subheader(f"{nombre}: {entrada} → {salida}")
                st.caption(f"Cuenta: (1 × {entrada}) · ({entrada} × {salida}) + "
                           f"(1 × {salida}) = 1 × {salida}  ·  "
                           f"{entrada * salida + salida:,} pesos y sesgos".replace(",", "."))
                et = list(generos) if es_salida else None
                st.pyplot(barras_con_signo(valores, f"Puntuación de cada neurona "
                                           f"({salida})", etiquetas=et, alto=2.6))
                entrada = salida
            else:
                apagadas = int((valores == 0).sum())
                st.markdown(f"**ReLU {n}:** {apagadas} de {len(valores)} neuronas "
                            f"apagadas para esta canción; {len(valores) - apagadas} "
                            f"se quedan activas.")
                fig, ax = plt.subplots(figsize=(8, 1.2))
                ax.imshow(valores[None, :], aspect="auto", cmap="Purples")
                ax.set_yticks([])
                ax.set_xlabel("neurona (blanco = apagada)")
                fig.tight_layout()
                st.pyplot(fig)
        with st.expander("¿Y los pesos? Las tablas de la red"):
            for i, capa in enumerate([c for c in red.capas if es_lineal(c)], 1):
                st.markdown(f"**W{i}:** {capa.W.shape[0]} × {capa.W.shape[1]} "
                            f"(filas = lo que entra, columnas = neuronas)  ·  "
                            f"**b{i}:** 1 × {capa.b.shape[0]}")

# ------------------------------------------------------------------- tab 5
with t5:
    if datos is None:
        st.info(SIN_CANCION)
    else:
        Z, P = datos["Z"], datos["P"]
        mejor = int(P.argmax())
        paso("<b>Qué ves:</b> la capa de salida da 10 puntuaciones sueltas (pueden ser "
             "negativas). El <b>softmax</b> las convierte en 10 porcentajes que suman "
             "100 %: se calcula <i>e elevado a cada puntuación</i> y se divide cada "
             "uno entre la suma de todos. Gana el género con el porcentaje más alto.")
        st.markdown(f"""
<div class="ganador">
  <p class="etiqueta">Parece</p>
  <p class="genero">{ICONOS.get(generos[mejor], '🎵')} {generos[mejor]}</p>
  <p class="confianza">con un {P[mejor]:.0%} de probabilidad</p>
</div>""", unsafe_allow_html=True)
        st.write("")
        E = np.exp(Z - Z.max())
        tabla = pd.DataFrame({"género": generos, "puntuación Z": Z.round(2),
                              "e^(Z − máx)": E.round(3),
                              "porcentaje": (P * 100).round(1)})
        orden = np.argsort(P)[::-1]
        c1, c2 = st.columns([3, 2])
        with c1:
            fig, ax = plt.subplots(figsize=(7, 3.6))
            colores = [MORADO if i == mejor else GRIS for i in orden]
            ax.barh([generos[i] for i in orden], [P[i] * 100 for i in orden], color=colores)
            ax.invert_yaxis()
            ax.set_xlabel("probabilidad (%)")
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig)
        with c2:
            st.dataframe(tabla.iloc[orden], hide_index=True)
        st.caption("El softmax nunca da exactamente 100 %. Si el porcentaje más alto "
                   "es bajo, la red no está segura: la canción se parece a varios géneros.")
        if P[mejor] < 0.5:
            st.warning("La red duda: el ganador no llega al 50 %.")

# ------------------------------------------------------------------- tab 6
@st.cache_data
def evaluar_modelo():
    d = np.load("rasgos.npz", allow_pickle=False)
    X, y = d["X"], d["y"]
    (Xtr, ytr), (Xva, yva), (Xte, yte) = dividir(X, y)
    Xtr_n, Xva_n, Xte_n = (normalizar(A, mu, sigma) for A in (Xtr, Xva, Xte))
    M = matriz_confusion(yte, red.predecir(Xte_n), C)
    acc_red = np.trace(M) / M.sum()
    lineal = crear_mlp(40, [], C)
    entrenar(lineal, Adam(lr=1e-3), Xtr_n, ytr, Xva_n, yva, epocas=300,
             l2=1e-4, paciencia=20)
    acc_lin = (lineal.predecir(Xte_n) == yte).mean()
    return M, acc_red, acc_lin, len(ytr), len(yva), len(yte)


with t6:
    if not os.path.exists("rasgos.npz"):
        st.info("Falta rasgos.npz para calcular la evaluación.")
    else:
        paso("<b>Qué ves:</b> el modelo se evalúa con las canciones de <b>test</b>, "
             "que no se han usado ni para entrenar ni para decidir cuándo parar. "
             "Se miran una sola vez.")
        with st.spinner("Calculando métricas..."):
            M, acc_red, acc_lin, ntr, nva, nte = evaluar_modelo()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Acierto de la red (test)", f"{acc_red:.1%}")
        c2.metric("Modelo lineal (sin capas ocultas)", f"{acc_lin:.1%}")
        c3.metric("Azar", f"{1 / C:.0%}")
        c4.metric("Canciones", f"{ntr} / {nva} / {nte}",
                  help="entrenamiento / validación / test")
        st.caption(f"Con solo {nte} canciones de test y una única división, la "
                   "diferencia entre la red y el modelo lineal puede ser ruido: "
                   "se mueve varios puntos según el reparto.")
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown("**Matriz de confusión** (filas = género real, columnas = "
                        "lo que dijo la red)")
            fig, ax = plt.subplots(figsize=(6.5, 5.6))
            ax.imshow(M, cmap="Purples")
            ax.set_xticks(range(C))
            ax.set_xticklabels(generos, rotation=45, ha="right")
            ax.set_yticks(range(C))
            ax.set_yticklabels(generos)
            for i in range(C):
                for j in range(C):
                    ax.text(j, i, M[i, j], ha="center", va="center",
                            color="white" if M[i, j] > M.max() / 2 else "black")
            ax.set_xlabel("predicho")
            ax.set_ylabel("real")
            fig.tight_layout()
            st.pyplot(fig)
        with c2:
            p, r, f1 = precision_recall_f1(M)
            st.markdown("**Por género**")
            st.dataframe(pd.DataFrame({"género": generos, "precisión": p.round(2),
                                       "recall": r.round(2), "F1": f1.round(2)}),
                         hide_index=True)
        if os.path.exists("curvas.png"):
            st.markdown("**Curvas de entrenamiento** (error de entrenamiento y de "
                        "validación por época)")
            st.image("curvas.png", width=520)
        st.markdown("""
**Cómo leerlo**
- **Accuracy:** porcentaje de canciones bien clasificadas.
- **Precisión de un género:** de las que la red dijo que eran de ese género, cuántas lo eran.
- **Recall:** de las que eran de ese género, cuántas encontró.
- **F1:** nota que combina precisión y recall.
""")
