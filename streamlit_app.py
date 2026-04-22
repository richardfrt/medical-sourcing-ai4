from __future__ import annotations
import sys
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ModuleNotFoundError:
    pass
"""
Streamlit UI para búsqueda y alternativas equivalentes de dispositivos médicos GUDID.

Requisitos:
    pip install streamlit chromadb openai pandas

Ejecutar:
    streamlit run streamlit_app.py

Depende de:
    - ChromaDB ya indexada con `gudid_embeddings.py index --csv gudid_filtrado.csv`.
    - Variable de entorno OPENAI_API_KEY configurada.

Nota:
    El GUDID no incluye precios. Se simulan precios deterministas a partir del id del
    dispositivo para ilustrar el cálculo de ahorro. Puedes editarlos manualmente en la tabla.
"""

import hashlib
import os
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from gudid_embeddings import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_COLLECTION,
    DEFAULT_DB,
    DEFAULT_MODEL,
    _embed_many,
    _get_collection,
    _get_openai_client,
    comparar_dispositivos,
)


st.set_page_config(
    page_title="Dispositivos médicos GUDID · Alternativas",
    page_icon="🩺",
    layout="wide",
)


# ---------- Caching / recursos ----------


@st.cache_resource(show_spinner=False)
def get_resources(db_path: str, collection_name: str):
    coll = _get_collection(db_path, collection_name)
    client = _get_openai_client()
    return coll, client


@st.cache_data(show_spinner=False)
def embed_query(_client, query: str, model: str) -> List[float]:
    return _embed_many(_client, [query], model, batch=1)[0]


@st.cache_data(show_spinner=False)
def semantic_search(
    _coll,
    _client,
    query: str,
    k: int,
    embed_model: str,
) -> List[Dict[str, Any]]:
    vec = embed_query(_client, query, embed_model)
    res = _coll.query(
        query_embeddings=[vec],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    ids = (res.get("ids") or [[]])[0]
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    out: List[Dict[str, Any]] = []
    for i, doc, meta, dist in zip(ids, docs, metas, dists):
        meta = dict(meta or {})
        out.append(
            {
                "id": i,
                "nombre_comercial": meta.get("nombre_comercial", ""),
                "fabricante": meta.get("fabricante", ""),
                "gmdn": meta.get("gmdn", ""),
                "descripcion": meta.get("descripcion") or doc or "",
                "similitud": round(1.0 - float(dist), 4),
            }
        )
    return out


@st.cache_data(show_spinner=False)
def comparar_cached(dispositivo_a: Dict[str, str], dispositivo_b: Dict[str, str], chat_model: str) -> Dict[str, Any]:
    return comparar_dispositivos(dispositivo_a, dispositivo_b, chat_model=chat_model)


# ---------- Precio simulado determinista ----------


def precio_simulado(device_id: str, minimo: float = 40.0, maximo: float = 1800.0) -> float:
    """Precio reproducible derivado del id del dispositivo (solo ilustrativo)."""
    h = hashlib.sha256(device_id.encode("utf-8")).digest()
    n = int.from_bytes(h[:4], "big") / 2**32
    return round(minimo + n * (maximo - minimo), 2)


# ---------- UI: sidebar ----------


with st.sidebar:
    st.markdown("### Conexión")
    db_path = st.text_input("Ruta ChromaDB", value=DEFAULT_DB)
    collection_name = st.text_input("Colección", value=DEFAULT_COLLECTION)
    embed_model = st.text_input("Modelo de embeddings", value=DEFAULT_MODEL)
    chat_model = st.text_input("Modelo de chat", value=DEFAULT_CHAT_MODEL)
    k_busqueda = st.slider("Resultados del buscador", 3, 25, 10)
    k_alternativas = st.slider("Nº de alternativas", 3, 20, 8)
    calcular_equivalencia = st.checkbox(
        "Calcular % de equivalencia técnica (GPT-4o)",
        value=True,
        help="Puede tardar varios segundos por alternativa.",
    )
    if not os.environ.get("OPENAI_API_KEY"):
        st.warning("OPENAI_API_KEY no está definida en el entorno.")


# ---------- UI: encabezado ----------


st.title("Dispositivos médicos · Alternativas equivalentes")
st.caption(
    "Búsqueda semántica en GUDID + equivalencia técnica con GPT-4o. "
    "Los precios son simulados (deterministas) y editables."
)


# ---------- Recursos ----------

try:
    coll, oa_client = get_resources(db_path, collection_name)
except Exception as err:
    st.error(f"No se pudo abrir ChromaDB en '{db_path}' · '{collection_name}': {err}")
    st.stop()


# ---------- Layout principal ----------


col_izq, col_der = st.columns([0.42, 0.58], gap="large")


# ------- Columna izquierda: buscador y selección -------


with col_izq:
    st.subheader("1. Buscar producto")
    query = st.text_input(
        "Consulta semántica",
        value="stent para arteria coronaria",
        placeholder="Ej: catéter balón de angioplastia 3.0 mm",
    )

    if not query.strip():
        st.info("Escribe una consulta para empezar.")
        st.stop()

    try:
        resultados = semantic_search(coll, oa_client, query.strip(), k_busqueda, embed_model)
    except Exception as err:
        st.error(f"Error en la búsqueda: {err}")
        st.stop()

    if not resultados:
        st.warning("Sin resultados.")
        st.stop()

    etiquetas = [
        f"{r['nombre_comercial'] or '(sin nombre)'} · {r['fabricante'] or 's/ fabricante'} · sim={r['similitud']}"
        for r in resultados
    ]
    idx = st.radio(
        "Selecciona el producto de referencia",
        options=list(range(len(resultados))),
        format_func=lambda i: etiquetas[i],
        index=0,
    )
    ref = resultados[idx]

    st.markdown("#### Producto de referencia")
    st.markdown(
        f"**{ref['nombre_comercial'] or '(sin nombre)'}**  \n"
        f"*{ref['fabricante'] or 's/ fabricante'}*  \n"
        f"GMDN: `{ref['gmdn'] or '-'}`  \n"
        f"ID: `{ref['id']}`"
    )
    with st.expander("Descripción técnica"):
        st.write(ref["descripcion"] or "—")

    precio_ref_defecto = precio_simulado(ref["id"])
    precio_ref = st.number_input(
        "Precio del producto de referencia (€)",
        min_value=0.0,
        value=float(precio_ref_defecto),
        step=1.0,
        format="%.2f",
        help="Precio simulado; edítalo para reflejar tu caso real.",
    )


# ------- Columna derecha: alternativas -------


with col_der:
    st.subheader("2. Alternativas sugeridas")

    candidatos = [r for r in resultados if r["id"] != ref["id"]][: k_alternativas]
    if len(candidatos) < k_alternativas:
        try:
            extra = semantic_search(
                coll, oa_client, ref["descripcion"] or query, k_alternativas + 2, embed_model
            )
            for e in extra:
                if e["id"] != ref["id"] and all(e["id"] != c["id"] for c in candidatos):
                    candidatos.append(e)
                if len(candidatos) >= k_alternativas:
                    break
        except Exception:
            pass

    if not candidatos:
        st.info("No hay alternativas en la base de datos.")
        st.stop()

    filas: List[Dict[str, Any]] = []
    progress = st.progress(0.0, text="Evaluando alternativas...") if calcular_equivalencia else None

    for i, cand in enumerate(candidatos, start=1):
        precio_alt = precio_simulado(cand["id"])
        ahorro_abs = round(precio_ref - precio_alt, 2)
        ahorro_pct = round(100.0 * ahorro_abs / precio_ref, 1) if precio_ref > 0 else 0.0

        equiv_pct: Any = None
        resumen = ""
        diferencias = ""
        if calcular_equivalencia:
            try:
                cmp_res = comparar_cached(ref, cand, chat_model)
                equiv_pct = cmp_res.get("porcentaje_compatibilidad")
                resumen = cmp_res.get("resumen") or ""
                difs = cmp_res.get("diferencias") or []
                diferencias = " · ".join(str(d) for d in difs[:3])
            except Exception as err:  # noqa: BLE001
                equiv_pct = None
                resumen = f"Error al comparar: {err}"

            if progress is not None:
                progress.progress(i / len(candidatos), text=f"Evaluando alternativas... ({i}/{len(candidatos)})")

        filas.append(
            {
                "Nombre comercial": cand["nombre_comercial"],
                "Fabricante": cand["fabricante"],
                "GMDN": cand["gmdn"],
                "Similitud semántica": cand["similitud"],
                "% Equivalencia técnica": equiv_pct,
                "Precio estimado (€)": precio_alt,
                "Ahorro (€)": ahorro_abs,
                "Ahorro (%)": ahorro_pct,
                "Resumen": resumen,
                "Diferencias clave": diferencias,
                "ID": cand["id"],
            }
        )

    if progress is not None:
        progress.empty()

    df = pd.DataFrame(filas)
    if "% Equivalencia técnica" in df.columns and df["% Equivalencia técnica"].notna().any():
        df = df.sort_values(by=["% Equivalencia técnica", "Ahorro (€)"], ascending=[False, False]).reset_index(drop=True)
    else:
        df = df.sort_values(by=["Similitud semántica", "Ahorro (€)"], ascending=[False, False]).reset_index(drop=True)

    col_config = {
        "Similitud semántica": st.column_config.ProgressColumn(
            "Similitud semántica", min_value=0.0, max_value=1.0, format="%.2f"
        ),
        "% Equivalencia técnica": st.column_config.ProgressColumn(
            "% Equivalencia técnica", min_value=0, max_value=100, format="%d%%"
        ),
        "Precio estimado (€)": st.column_config.NumberColumn(format="%.2f €"),
        "Ahorro (€)": st.column_config.NumberColumn(format="%.2f €"),
        "Ahorro (%)": st.column_config.NumberColumn(format="%.1f %%"),
        "Resumen": st.column_config.TextColumn("Resumen", width="large"),
        "Diferencias clave": st.column_config.TextColumn("Diferencias clave", width="large"),
    }

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config=col_config,
    )

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Descargar alternativas (CSV)",
        data=csv_bytes,
        file_name="alternativas_sugeridas.csv",
        mime="text/csv",
    )

    st.caption(
        "Similitud: coseno embeddings. % Equivalencia técnica: GPT-4o sobre descripción GUDID. "
        "Precios simulados deterministas a partir del id del dispositivo."
    )
