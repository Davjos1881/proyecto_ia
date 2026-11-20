"""Demo del proyecto de IA: priorización de siniestros viales en Cali.

Cómo ejecutarla (desde la carpeta demo_siniestros):
    pip install -r requirements.txt
    streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
MODELO_DIR = BASE / "modelo"
DATOS_DIR = BASE / "datos"

st.set_page_config(page_title="Siniestros viales en Cali", layout="wide")

ETIQUETAS = {
    "veh_moto": "Motocicleta",
    "veh_motocarro": "Motocarro",
    "veh_automovil": "Automóvil",
    "veh_camioneta_campero": "Camioneta o campero",
    "veh_bus": "Bus",
    "veh_carga_pesada": "Vehículo de carga pesada",
    "veh_bicicleta": "Bicicleta",
}

# ---------------------------------------------------------------------------
# Resultados de los notebooks (los mismos que aparecen en el reporte)
# ---------------------------------------------------------------------------
ROC_AUC_MODELO_DEMO = 0.796  # modelo de solo vehículos, test 2022

CLASIFICACION = pd.DataFrame(
    {
        "Modelo": [
            "MLP (red neuronal)",
            "Random Forest",
            "Árbol de decisión",
            "Regresión logística",
            "SVM lineal",
            "k-NN (k = 15)",
            "Perceptrón",
        ],
        "ROC-AUC": [0.802, 0.798, 0.795, 0.793, 0.792, 0.786, 0.733],
        "PR-AUC": [0.744, 0.743, 0.726, 0.735, 0.734, 0.720, 0.681],
    }
)

CALENDARIO = pd.DataFrame(
    {
        "Modelo": [
            "Promedio del entrenamiento",
            "Promedio de los últimos 28 días",
            "Regresión lineal con calendario",
            "SGDRegressor con calendario",
        ],
        "Error medio (MAE)": [2.950, 2.913, 3.154, 3.160],
        "R²": [-0.019, -0.023, -0.180, -0.187],
    }
)

COBERTURA = pd.DataFrame(
    {
        "Cruces en la lista": [50, 100, 200],
        "Lista de 2016–2017, graves de 2021–2022": [10.6, 16.3, 24.0],
        "Lista de 2016–2022, graves de 2023–2024": [5.3, 7.5, 11.3],
    }
).set_index("Cruces en la lista")


# ---------------------------------------------------------------------------
# Carga de archivos
# ---------------------------------------------------------------------------
@st.cache_resource
def cargar_modelo():
    modelo = joblib.load(MODELO_DIR / "mlp_vehiculos.joblib")
    columnas = json.loads((MODELO_DIR / "columnas.json").read_text(encoding="utf-8"))
    return modelo, columnas


@st.cache_data
def cargar_csv(nombre):
    ruta = DATOS_DIR / nombre
    return pd.read_csv(ruta) if ruta.exists() else None


def miles(n):
    return f"{int(n):,}".replace(",", ".")


# ---------------------------------------------------------------------------
# Pestaña 1: atender un reporte
# ---------------------------------------------------------------------------
def etiqueta_combinacion(fila, columnas):
    partes = [ETIQUETAS.get(c, c) for c in columnas if int(fila[c]) == 1]
    return " + ".join(partes) if partes else "Sin vehículo o sin dato"


def pestana_respuesta():
    st.subheader("¿A qué reporte atender primero?")
    st.write(
        "Elige los vehículos que se informan en la llamada. El modelo los compara "
        "con el historial y marca el nivel de prioridad."
    )

    try:
        modelo, columnas = cargar_modelo()
    except FileNotFoundError:
        st.error(
            "No se encontró el modelo. Revisa que existan modelo/mlp_vehiculos.joblib "
            "y modelo/columnas.json junto a app.py."
        )
        return

    tasa = cargar_csv("tasa_combinaciones.csv")
    nombre_a_col = {ETIQUETAS.get(c, c): c for c in columnas}
    por_defecto = [n for n in ("Motocicleta", "Automóvil") if n in nombre_a_col]

    elegidos = st.multiselect(
        "Vehículos involucrados", list(nombre_a_col), default=por_defecto
    )
    st.caption("Cuenta qué tipos hay, no cuántos: dos automóviles equivalen a uno.")

    if not elegidos:
        st.info(
            "Elige al menos un vehículo. Sin vehículos, el resultado mezcla "
            "“ninguno” con “sin dato”."
        )
        return

    sel = {nombre_a_col[n] for n in elegidos}
    x = pd.DataFrame([{c: int(c in sel) for c in columnas}], columns=columnas)
    puntaje = float(modelo.predict_proba(x)[0, 1])

    base, hist, n_hist = 0.40, None, 0
    if tasa is not None:
        base = float((tasa["n"] * tasa["pct_grave"]).sum() / tasa["n"].sum())
        mask = np.ones(len(tasa), dtype=bool)
        for c in columnas:
            mask &= tasa[c].astype(int).to_numpy() == int(c in sel)
        if mask.any():
            fila = tasa[mask].iloc[0]
            hist, n_hist = float(fila["pct_grave"]), int(fila["n"])

    # El puntaje no es una probabilidad calibrada: el nivel se define frente al promedio histórico.
    if puntaje >= base + 0.10:
        st.error(
            "**Prioridad alta.** Con estos vehículos, el modelo ve más riesgo de "
            "lesionados o fallecidos que el promedio histórico."
        )
    elif puntaje <= base - 0.10:
        st.success(
            "**Prioridad baja.** Con estos vehículos, el modelo ve menos riesgo de "
            "lesionados o fallecidos que el promedio histórico."
        )
    else:
        st.warning(
            "**Prioridad media.** El riesgo estimado es cercano al promedio histórico."
        )

    c1, c2, c3 = st.columns(3)
    if hist is not None:
        c1.metric(
            "Graves con esta combinación",
            f"{hist:.0%}",
            help=f"{miles(n_hist)} siniestros de 2016 a 2022 con estos mismos vehículos.",
        )
    else:
        c1.metric("Graves con esta combinación", "Sin datos")
    c2.metric(
        "Graves en todos los siniestros",
        f"{base:.0%}",
        help="Promedio histórico de siniestros con lesionados o fallecidos.",
    )
    c3.metric(
        "Puntaje del modelo",
        f"{puntaje:.2f}",
        help="Sirve para ordenar casos. No es una probabilidad calibrada.",
    )
    if hist is not None and n_hist < 30:
        st.caption("Hay pocos casos históricos con esta combinación: tómalo con cuidado.")

    if tasa is not None:
        st.markdown("**Combinaciones más frecuentes en el historial**")
        top = tasa.sort_values("n", ascending=False).head(10).copy()
        top["Combinación"] = top.apply(lambda f: etiqueta_combinacion(f, columnas), axis=1)
        vista = pd.DataFrame(
            {
                "Combinación": top["Combinación"],
                "Siniestros": top["n"].astype(int),
                "% graves": (top["pct_grave"] * 100).round(0).astype(int),
            }
        )
        st.dataframe(vista, hide_index=True)

    st.caption(
        "Supone que el tipo de vehículos se conoce al recibir el reporte, algo que aún "
        "no está verificado. Apoya la decisión de quien despacha, no la reemplaza."
    )


# ---------------------------------------------------------------------------
# Pestaña 2: dónde poner recursos
# ---------------------------------------------------------------------------
def nombrar_grupos(df):
    ids = sorted(df["cluster"].unique())
    if len(ids) != 3:
        return {c: f"Grupo {c}" for c in ids}
    medias = df.groupby("cluster")[["volumen_total", "pct_graves"]].mean()
    mas_volumen = medias["volumen_total"].idxmax()
    resto = medias.drop(index=mas_volumen)
    mas_grave = resto["pct_graves"].idxmax()
    nombres = {mas_volumen: "Muchos siniestros", mas_grave: "Más graves y más motos"}
    for c in resto.index:
        nombres.setdefault(c, "Menos graves, más carga pesada")
    return nombres


def preparar_cruces(df):
    df = df.copy()
    if "graves" not in df.columns:
        df["graves"] = (df["volumen_total"] * df["pct_graves"]).round().astype(int)
    if "cluster" in df.columns:
        df["grupo"] = df["cluster"].map(nombrar_grupos(df))
    else:
        df["grupo"] = "Sin grupo"
    return df


def pestana_cruces():
    st.subheader("¿Dónde poner recursos fijos?")
    st.write(
        "Los cruces con más siniestros graves en años anteriores siguen concentrando "
        "siniestros graves después. Esta lista sirve para priorizar."
    )

    st.markdown("**¿Cuántos cruces priorizarías?**")
    n = st.select_slider("Cruces en la lista", options=[50, 100, 200], value=100)
    col_a, col_b = st.columns(2)
    col_a.metric(
        "Graves cubiertos en 2021–2022",
        f"{COBERTURA.loc[n].iloc[0]:.1f}%",
        help="Lista elegida con 2016–2017.",
    )
    col_b.metric(
        "Graves cubiertos en 2023–2024",
        f"{COBERTURA.loc[n].iloc[1]:.1f}%",
        help="Lista elegida con 2016–2022.",
    )
    st.bar_chart(COBERTURA)
    st.caption(
        "Porcentaje de los graves posteriores, con dirección registrada, que ocurre en "
        "la lista. Por azar se esperaría cerca de 1%. Más del 75% de los graves ocurre "
        "fuera de la lista."
    )

    st.markdown("**Los cruces**")
    cruces = cargar_csv("cruces_perfil.csv")
    if cruces is None:
        st.info(
            "Falta el archivo datos/cruces_perfil.csv. Expórtalo desde el notebook 3 "
            "con perfil_cruces.to_csv(...)."
        )
        return
    df = preparar_cruces(cruces)

    f1, f2, f3 = st.columns([2, 2, 1])
    grupos = sorted(df["grupo"].unique())
    elegidos = f1.multiselect("Grupo", grupos, default=grupos)
    texto = f2.text_input("Buscar calle o carrera", "")
    min_graves = f3.number_input("Mínimo de graves", 0, int(df["graves"].max()), 0)

    vista_df = df[df["grupo"].isin(elegidos) & (df["graves"] >= min_graves)]
    if texto.strip():
        vista_df = vista_df[
            vista_df["lugar"].str.contains(texto.strip(), case=False, regex=False, na=False)
        ]
    vista_df = vista_df.sort_values("graves", ascending=False)

    total_lista = int(df["graves"].sum())
    m1, m2, m3 = st.columns(3)
    m1.metric("Cruces mostrados", miles(len(vista_df)))
    m2.metric("Graves en esos cruces", miles(vista_df["graves"].sum()))
    m3.metric(
        "Parte de los graves de la lista",
        f"{vista_df['graves'].sum() / total_lista:.0%}" if total_lista else "-",
    )

    tabla = pd.DataFrame(
        {
            "Cruce": vista_df["lugar"],
            "Siniestros": vista_df["volumen_total"].round().astype(int),
            "Graves": vista_df["graves"].astype(int),
            "% graves": (vista_df["pct_graves"] * 100).round(0).astype(int),
        }
    )
    if "pct_motos" in vista_df.columns:
        tabla["% motos"] = (vista_df["pct_motos"] * 100).round(0).astype(int)
    if "pct_carga_pesada" in vista_df.columns:
        tabla["% carga pesada"] = (vista_df["pct_carga_pesada"] * 100).round(0).astype(int)
    tabla["Grupo"] = vista_df["grupo"]
    st.dataframe(tabla, hide_index=True)

    st.caption(
        "Los grupos son una segmentación descriptiva, no categorías rígidas. "
        "“Más motos” indica que hubo una moto involucrada, no que el herido fuera el "
        "motociclista. Solo se usan siniestros con dirección válida (2016–2017 y 2021–2022)."
    )


# ---------------------------------------------------------------------------
# Pestaña 3: evidencia y límites
# ---------------------------------------------------------------------------
def pestana_evidencia():
    st.subheader("Qué sostiene cada resultado")

    st.markdown("**1. Clasificar la gravedad de un siniestro**")
    comparacion = cargar_csv("comparacion_modelos.csv")
    if comparacion is not None and {"Modelo", "ROC-AUC", "PR-AUC"} <= set(comparacion.columns):
        tabla = comparacion[["Modelo", "ROC-AUC", "PR-AUC"]]
    else:
        tabla = CLASIFICACION
    st.dataframe(tabla, hide_index=True)
    st.bar_chart(tabla.set_index("Modelo")["ROC-AUC"])
    st.write(
        "Cinco de los siete modelos quedan entre 0.79 y 0.80, una diferencia que un solo "
        "año de prueba no permite separar. La gravedad depende sobre todo de qué "
        "vehículos participan. El modelo de esta demo usa solo los vehículos y llega a "
        f"{ROC_AUC_MODELO_DEMO:.3f} en el test de 2022."
    )
    with st.expander("¿Qué es el ROC-AUC?"):
        st.write(
            "Es la probabilidad de que el modelo le asigne más riesgo a un siniestro grave "
            "que a uno de solo daños. 0.5 es azar y 1 es perfecto. No depende del punto de "
            "corte, por eso sirve para comparar modelos aunque la proporción de graves "
            "cambie entre años."
        )

    st.markdown("**2. ¿El calendario ayuda a predecir cuántos graves habrá por día?**")
    st.dataframe(CALENDARIO, hide_index=True)
    st.write(
        "No. Con el calendario el error es mayor que con el simple promedio. Los festivos "
        "tienen cerca de 20% menos graves, pero el efecto es pequeño frente a la "
        "variación diaria. Prueba: entrenamiento 2020–2022, test 2023."
    )

    st.markdown("**3. Límites del proyecto**")
    st.markdown(
        "- Solo cerca del 37% de los siniestros de 2016–2022 tiene dirección válida, y "
        "ninguno entre 2018 y 2020: el análisis de cruces no cubre toda la ciudad.\n"
        "- En el 18.6% de los siniestros no hay dato de vehículos, y no se sabe si ese "
        "dato está disponible al recibir el reporte.\n"
        "- El registro cambió con el tiempo: los siniestros de solo daños casi "
        "desaparecen desde 2023.\n"
        "- “Grave” significa con lesionados o fallecidos, no necesariamente crítico.\n"
        "- La clasificación se evaluó con un solo año de prueba (2022).\n"
        "- No hay datos de tiempos de respuesta ni del efecto de intervenir un cruce: "
        "no se puede afirmar cuánto tiempo o cuántos siniestros se evitarían."
    )


# ---------------------------------------------------------------------------
# Página
# ---------------------------------------------------------------------------
st.title("Siniestros viales en Cali: priorización")
st.write(
    "Demo del proyecto de IA con datos de siniestralidad de 2016 a 2024. Responde dos "
    "preguntas: a qué reporte atender primero y dónde conviene poner recursos fijos."
)

tab1, tab2, tab3 = st.tabs(["Atender un reporte", "Dónde poner recursos", "Evidencia y límites"])
with tab1:
    pestana_respuesta()
with tab2:
    pestana_cruces()
with tab3:
    pestana_evidencia()