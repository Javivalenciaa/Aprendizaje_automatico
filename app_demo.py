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
from datos.audio import cargar_centro, rasgos_segmentos, espectrograma
from datos.preparar import cargar_segmentos, normalizar
from datos.metricas import matriz_confusion, precision_recall_f1, votar_por_cancion

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
    red = crear_mlp(len(d["mu"]), d["ocultas"].tolist(), len(generos))
    cargar_pesos(red, "modelo.npz")
    return red, d["mu"], d["sigma"], generos, d["ocultas"].tolist()


red, mu, sigma, generos, ocultas = cargar_modelo()
C = len(generos)
n_pesos = sum(p.size for p, _ in red.params())


def es_lineal(capa):
    return hasattr(capa, "W")


# ---------------------------------------------------------------- analisis
def analizar(contenido):
    """Carga 30 s del centro de la cancion, la corta en trozos de 3 s y saca
    43 rasgos de cada trozo (40 de timbre y 3 de ritmo)."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(contenido)
        ruta = f.name
    senal, sr = cargar_centro(ruta)
    os.remove(ruta)
    S = rasgos_segmentos(senal, sr)                 # (trozos x 43)
    X = normalizar(S, mu, sigma)
    P = red.predecir_proba(X)                       # probabilidades por trozo
    return dict(senal=senal, sr=sr, S=S, X=X, P=P, n=len(S))


def recorrido(x):
    """Pasa UN trozo por la red guardando el resultado de cada capa."""
    etapas, h, n_lin = [], x[None, :], 0
    for capa in red.capas:
        h = capa.forward(h)
        if es_lineal(capa):
            n_lin += 1
            etapas.append(("lineal", n_lin, h[0].copy(), capa))
        else:
            etapas.append(("relu", n_lin, h[0].copy(), capa))
    return etapas, h[0]


def mfcc_trozo(datos, k):
    import librosa
    n = 3 * datos["sr"]
    t = datos["senal"][k * n:(k + 1) * n]
    t = t / (np.sqrt(np.mean(t ** 2)) + 1e-9)
    return librosa.feature.mfcc(y=t, sr=datos["sr"], n_mfcc=21)[1:]


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
    arq = " → ".join([str(len(mu))] + [str(o) for o in ocultas] + [str(C)])
    st.markdown(f"**Red:** {arq}  \n**Pesos y sesgos:** {n_pesos:,}".replace(",", "."))
    st.markdown("**Géneros:**  \n" +
                "  \n".join(f"{ICONOS.get(g, '🎵')} {g}" for g in generos))
    st.caption("Solo acepta .wav. Analiza 30 s del centro de la canción y "
               "siempre elige uno de estos géneros. Entrenado con GTZAN "
               "(música de los 90-2000): con música moderna duda más.")

st.title("🎶 Del audio a la predicción")
st.write("Sube una canción y sigue, paso a paso, lo que hace el modelo "
         "hasta decidir su género.")

archivo = st.file_uploader("Sube una canción (.wav)", type=["wav"])
datos, k = None, 0
if archivo is not None:
    with st.spinner("Analizando la canción..."):
        datos = analizar(archivo.getvalue())
    k = st.slider("Trozo de 3 s que quieres seguir paso a paso (pestañas 1-5)",
                  1, datos["n"], 1) - 1

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

`audio` → `30 s del centro` → `10 trozos de 3 s` → `43 rasgos por trozo (timbre + ritmo)` →
`normalizar` → `capa 1 + ReLU` → `capa 2 + ReLU` → `capa de salida` →
`softmax` → `media de los 10 trozos` → `género`
""")
    else:
        st.audio(archivo)
        senal, sr = datos["senal"], datos["sr"]
        paso(f"<b>Qué ves:</b> un audio es una lista enorme de números: estos 30 s son "
             f"<b>{len(senal):,}</b> números ({sr} por segundo). Se cogen del <b>centro</b> de "
             f"la canción (la intro engaña) y se cortan en <b>{datos['n']} trozos de 3 s</b>. "
             f"Cada trozo es una mini-canción que la red clasificará por separado; al final "
             f"se juntan sus votos.".replace(",", "."))
        fig, ax = plt.subplots(figsize=(10, 2.2))
        ax.plot(np.arange(len(senal)) / sr, senal, color=MORADO, lw=.4)
        ax.axvspan(k * 3, (k + 1) * 3, color="#ffd54f", alpha=.45, label="trozo elegido")
        for i in range(1, datos["n"]):
            ax.axvline(i * 3, color="#444", lw=.6, ls=":")
        ax.set_xlabel("tiempo (s)")
        ax.set_ylabel("amplitud")
        ax.legend(loc="upper right")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        st.markdown("**Espectrograma:** cuánta energía hay en graves y agudos a lo largo del tiempo.")
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
        paso("<b>Qué ves:</b> de cada trozo se sacan <b>20 números (MFCC)</b> que describen la "
             "forma de su espectro, de lo general a lo fino. Para cada MFCC se calcula la "
             "<b>media</b> y la <b>desviación</b> a lo largo del trozo: 20 + 20 = <b>40 rasgos de timbre</b>. "
             "Antes se <b>iguala el volumen</b> del trozo y se descarta el MFCC 0, que solo mide "
             "lo fuerte que suena. A esos 40 se suman <b>3 rasgos de ritmo</b>: el tempo (pulsos "
             "por minuto) y cuánta actividad rítmica hay (media y desviación). Total: <b>43 rasgos</b>.")
        mf = mfcc_trozo(datos, k)
        fig, ax = plt.subplots(figsize=(10, 2.8))
        im = ax.imshow(mf, origin="lower", aspect="auto", cmap="RdBu_r")
        ax.set_xlabel(f"instante dentro del trozo {k + 1} (→)")
        ax.set_ylabel("MFCC nº")
        fig.colorbar(im, ax=ax, fraction=.03)
        fig.tight_layout()
        st.pyplot(fig)
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(barras_con_signo(datos["S"][k][:20], "20 rasgos: MEDIA de cada MFCC"))
        with c2:
            st.pyplot(barras_con_signo(datos["S"][k][20:], "20 rasgos: DESVIACIÓN de cada MFCC"))
        r1, r2, r3 = st.columns(3)
        r1.metric("Tempo de la canción", f"{datos['S'][k][42]:.0f} BPM")
        r2.metric("Actividad rítmica (media)", f"{datos['S'][k][40]:.2f}")
        r3.metric("Actividad rítmica (desviación)", f"{datos['S'][k][41]:.2f}")
        st.markdown("**Todos los trozos:** cada fila es un trozo de 3 s y cada columna un rasgo.")
        fig, ax = plt.subplots(figsize=(10, 2.6))
        im = ax.imshow(datos["X"], aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)
        ax.set_yticks(range(datos["n"]))
        ax.set_yticklabels(range(1, datos["n"] + 1), fontsize=7)
        ax.set_xlabel("rasgo (0-19 medias · 20-39 desviaciones · 40-42 ritmo)")
        ax.set_ylabel("trozo")
        fig.colorbar(im, ax=ax, fraction=.03)
        fig.tight_layout()
        st.pyplot(fig)
        st.success(f"Una canción de 30 s se queda en una tabla de {datos['n']} × 43 números.")

# ------------------------------------------------------------------- tab 3
with t3:
    if datos is None:
        st.info(SIN_CANCION)
    else:
        paso("<b>Qué ves:</b> los 43 rasgos tienen escalas muy distintas. Se normalizan con la "
             "<b>media y la desviación de los trozos de entrenamiento</b> (guardadas en "
             "<code>modelo.npz</code>): <code>(x − media) / desviación</code>. Siempre las del "
             "entrenamiento, nunca unas calculadas con la canción nueva.")
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(barras_con_signo(datos["S"][k], "Antes: rasgos tal cual", alto=2.8))
        with c2:
            st.pyplot(barras_con_signo(datos["X"][k], "Después: rasgos normalizados", alto=2.8))
        nombres = [f"media MFCC {i + 1}" for i in range(20)] + \
                  [f"desv. MFCC {i + 1}" for i in range(20)] + \
                  ["actividad rítmica (media)", "actividad rítmica (desv.)", "tempo (BPM)"]
        tabla = pd.DataFrame({"rasgo": nombres, "valor": datos["S"][k],
                              "media del entrenamiento": mu,
                              "desviación del entrenamiento": sigma,
                              "normalizado": datos["X"][k]}).round(2)
        with st.expander("Ver la cuenta de los 43 rasgos"):
            st.dataframe(tabla, hide_index=True)

# ------------------------------------------------------------------- tab 4
with t4:
    if datos is None:
        st.info(SIN_CANCION)
    else:
        etapas, Z = recorrido(datos["X"][k])
        paso(f"<b>Qué ves (trozo {k + 1}):</b> la red aplica capa tras capa. Cada <b>capa lineal</b> "
             "calcula, para cada neurona, <code>entrada · pesos + sesgo</code>. Cada "
             "<b>ReLU</b> deja pasar los positivos y pone a 0 los negativos "
             "(neurona apagada). Verde = positivo, naranja = negativo.")
        entrada = len(mu)
        lineales = [c for c in red.capas if es_lineal(c)]
        for tipo, n, valores, capa in etapas:
            if tipo == "lineal":
                salida = len(valores)
                es_salida = capa is lineales[-1]
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
                            f"apagadas para este trozo; {len(valores) - apagadas} "
                            f"se quedan activas.")
                fig, ax = plt.subplots(figsize=(8, 1.2))
                ax.imshow(valores[None, :], aspect="auto", cmap="Purples")
                ax.set_yticks([])
                ax.set_xlabel("neurona (blanco = apagada)")
                fig.tight_layout()
                st.pyplot(fig)
        with st.expander("¿Y los pesos? Las tablas de la red"):
            for i, capa in enumerate(lineales, 1):
                st.markdown(f"**W{i}:** {capa.W.shape[0]} × {capa.W.shape[1]} "
                            f"(filas = lo que entra, columnas = neuronas)  ·  "
                            f"**b{i}:** 1 × {capa.b.shape[0]}")

# ------------------------------------------------------------------- tab 5
with t5:
    if datos is None:
        st.info(SIN_CANCION)
    else:
        etapas, Z = recorrido(datos["X"][k])
        Pk = datos["P"][k]
        P = datos["P"].mean(axis=0)
        mejor = int(P.argmax())
        paso(f"<b>Qué ves:</b> para el trozo {k + 1}, la capa de salida da 10 puntuaciones sueltas. "
             "El <b>softmax</b> las convierte en 10 porcentajes que suman 100 %: calcula "
             "<i>e elevado a cada puntuación</i> y divide cada uno entre la suma. Esto se hace "
             "en cada trozo; la canción se decide <b>promediando</b> los porcentajes de todos.")
        E = np.exp(Z - Z.max())
        tabla = pd.DataFrame({"género": generos, "puntuación Z": Z.round(2),
                              "e^(Z − máx)": E.round(3),
                              "porcentaje": (Pk * 100).round(1)})
        orden_k = np.argsort(Pk)[::-1]
        st.markdown(f"**Softmax del trozo {k + 1}**")
        st.dataframe(tabla.iloc[orden_k], hide_index=True)
        st.markdown("**Los trozos votan:** cada fila es un trozo; cada columna, un género. "
                    "Cuanto más oscuro, más seguro está ese trozo.")
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.imshow(datos["P"], aspect="auto", cmap="Purples", vmin=0, vmax=1)
        ax.set_xticks(range(C))
        ax.set_xticklabels(generos, rotation=40, ha="right")
        ax.set_yticks(range(datos["n"]))
        ax.set_yticklabels(range(1, datos["n"] + 1), fontsize=7)
        ax.set_ylabel("trozo")
        fig.tight_layout()
        st.pyplot(fig)
        st.markdown("**Resultado: media de todos los trozos**")
        st.markdown(f"""
<div class="ganador">
  <p class="etiqueta">Parece</p>
  <p class="genero">{ICONOS.get(generos[mejor], '🎵')} {generos[mejor]}</p>
  <p class="confianza">con un {P[mejor]:.0%} de probabilidad</p>
</div>""", unsafe_allow_html=True)
        st.write("")
        orden = np.argsort(P)[::-1]
        fig, ax = plt.subplots(figsize=(7, 3.4))
        colores = [MORADO if i == mejor else GRIS for i in orden]
        ax.barh([generos[i] for i in orden], [P[i] * 100 for i in orden], color=colores)
        ax.invert_yaxis()
        ax.set_xlabel("probabilidad (%)")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        st.caption("El softmax nunca da exactamente 100 %. Si el porcentaje más alto es bajo, "
                   "la red no está segura: la canción se parece a varios géneros.")
        if P[mejor] < 0.5:
            st.warning("La red duda: el ganador no llega al 50 %.")


# ------------------------------------------------------------------- tab 6
@st.cache_data
def evaluar_modelo():
    d = cargar_segmentos()
    Xtr, Xva, Xte = (normalizar(d[q], mu, sigma) for q in ("Xtr", "Xva", "Xte"))
    real, pred = votar_por_cancion(red.predecir_proba(Xte), d["cte"], d["yte"])
    M = matriz_confusion(real, pred, C)
    lineal = crear_mlp(len(mu), [], C)
    entrenar(lineal, Adam(lr=1e-3), Xtr, d["ytr"], Xva, d["yva"], epocas=300,
             l2=1e-3, paciencia=15)
    real, pred = votar_por_cancion(lineal.predecir_proba(Xte), d["cte"], d["yte"])
    return M, np.trace(M) / M.sum(), (real == pred).mean(), len(real), len(d["ytr"])


with t6:
    if not os.path.exists("rasgos_seg.npz"):
        st.info("Falta rasgos_seg.npz para calcular la evaluación.")
    else:
        paso("<b>Qué ves:</b> el modelo se evalúa con canciones de <b>test</b>, que no se han "
             "usado ni para entrenar ni para decidir cuándo parar. Cada canción puntúa una vez "
             "(media de sus trozos).")
        with st.spinner("Calculando métricas..."):
            M, acc_red, acc_lin, nte, ntr = evaluar_modelo()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Acierto de la red (test)", f"{acc_red:.1%}")
        c2.metric("Modelo lineal (sin capas ocultas)", f"{acc_lin:.1%}")
        c3.metric("Azar", f"{1 / C:.0%}")
        c4.metric("Canciones de test", f"{nte}")
        st.caption(f"Con solo {nte} canciones de test, cada punto de acierto son 1,5 canciones: "
                   "diferencias de 3-4 puntos pueden ser ruido. Entrenado con "
                   f"{ntr} trozos (originales y distorsionados).")
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
**Límites que conviene decir en voz alta**
- Se entrenó con GTZAN: 1.000 clips de 30 s de música de los 90-2000. La música moderna (otra producción, otros estilos) se parece menos a lo que vio y la red duda más.
- El rock es el género más difícil: la red lo reparte entre blues, pop, country y metal (solo acierta 2 de 15).
- Los 10 géneros son los únicos que conoce: siempre elige uno de ellos.

**Cómo leerlo**
- **Accuracy:** porcentaje de canciones bien clasificadas.
- **Precisión de un género:** de las que la red dijo que eran de ese género, cuántas lo eran.
- **Recall:** de las que eran de ese género, cuántas encontró.
- **F1:** nota que combina precisión y recall.
""")
