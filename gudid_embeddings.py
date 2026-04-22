from __future__ import annotations
import sys
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ModuleNotFoundError:
    pass

"""
Genera embeddings con OpenAI para las descripciones técnicas del CSV de GUDID
y las guarda en ChromaDB (persistente en disco) para búsquedas semánticas.

Requisitos:
    pip install openai chromadb

Uso típico:

    # 1) Indexa el CSV (lo generado por gudid_filter.py)
    python gudid_embeddings.py index --csv gudid_filtrado.csv

    # 2) Busca por significado, no por coincidencia exacta
    python gudid_embeddings.py search --query "stent para arteria coronaria" --k 5

Variables de entorno:
    OPENAI_API_KEY   Tu clave de OpenAI (obligatoria).

Notas:
    - Modelo por defecto: text-embedding-3-small (1536 dims, barato y rápido).
    - La BD se persiste por defecto en ./chroma_db (puedes cambiar con --db).
    - Se salta filas sin descripción y evita reindexar las ya existentes (usa --rebuild si quieres forzar).
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import chromadb
except ImportError as e:  # noqa: BLE001
    print("Falta la dependencia 'chromadb'. Instala con: pip install chromadb", file=sys.stderr)
    raise

try:
    from openai import OpenAI
except ImportError:
    print("Falta la dependencia 'openai'. Instala con: pip install openai", file=sys.stderr)
    raise


DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_DB = "./chroma_db"
DEFAULT_COLLECTION = "gudid_devices"
DEFAULT_BATCH = 64
DEFAULT_CHAT_MODEL = "gpt-4o"


COLUMN_ALIASES = {
    "nombre_comercial": ("nombre comercial", "brand_name", "brandname"),
    "descripcion": (
        "descripción técnica",
        "descripcion tecnica",
        "description",
        "device_description",
    ),
    "fabricante": ("fabricante", "company_name", "companyname"),
    "gmdn": (
        "código gmdn (categoría global)",
        "codigo gmdn",
        "gmdn_code",
        "gmdncode",
    ),
}


def _norm(h: str) -> str:
    return h.strip().lower().replace("_", " ")


def _resolve_headers(header: Sequence[str]) -> Dict[str, int]:
    hmap = {_norm(h): i for i, h in enumerate(header)}
    out: Dict[str, int] = {}
    for key, aliases in COLUMN_ALIASES.items():
        for a in aliases:
            if a in hmap:
                out[key] = hmap[a]
                break
    faltan = [k for k in COLUMN_ALIASES if k not in out]
    if faltan:
        raise RuntimeError(
            f"No se encontraron columnas: {faltan}. Cabecera detectada: {list(header)}"
        )
    return out


def _row_id(brand: str, desc: str, company: str) -> str:
    h = hashlib.sha1()
    h.update(brand.encode("utf-8"))
    h.update(b"\x00")
    h.update(company.encode("utf-8"))
    h.update(b"\x00")
    h.update(desc.encode("utf-8"))
    return h.hexdigest()[:24]


def _read_csv_rows(path: str) -> Tuple[List[str], List[List[str]]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [r for r in reader if r]
    return header, rows


def _batched(xs: Sequence, n: int) -> Iterable[Sequence]:
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


def _embed_many(
    client: OpenAI,
    texts: Sequence[str],
    model: str,
    batch: int,
) -> List[List[float]]:
    out: List[List[float]] = []
    for chunk in _batched(texts, batch):
        tries = 0
        while True:
            try:
                resp = client.embeddings.create(model=model, input=list(chunk))
                out.extend(d.embedding for d in resp.data)
                break
            except Exception as err:  # noqa: BLE001
                tries += 1
                if tries >= 5:
                    raise
                wait = min(2**tries, 30)
                print(
                    f"[warn] error de embeddings ({err}). Reintento {tries} en {wait}s...",
                    file=sys.stderr,
                )
                time.sleep(wait)
    return out


def _get_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Falta OPENAI_API_KEY en el entorno.", file=sys.stderr)
        sys.exit(2)
    return OpenAI(api_key=api_key)


def _get_collection(
    db_path: str,
    name: str,
    *,
    reset: bool = False,
):
    client = chromadb.PersistentClient(path=db_path)
    if reset:
        try:
            client.delete_collection(name)
        except Exception:  # noqa: BLE001
            pass
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def cmd_index(args: argparse.Namespace) -> int:
    header, rows = _read_csv_rows(args.csv)
    cols = _resolve_headers(header)

    ids: List[str] = []
    docs: List[str] = []
    metas: List[Dict[str, str]] = []

    for r in rows:
        def cell(k: str) -> str:
            i = cols[k]
            return (r[i] if i < len(r) else "").strip()

        desc = cell("descripcion")
        if not desc:
            continue
        brand = cell("nombre_comercial")
        company = cell("fabricante")
        gmdn = cell("gmdn")

        ids.append(_row_id(brand, desc, company))
        docs.append(desc)
        metas.append(
            {
                "nombre_comercial": brand,
                "fabricante": company,
                "gmdn": gmdn,
                "descripcion": desc,
            }
        )

    if not ids:
        print("No hay filas con descripción para indexar.", file=sys.stderr)
        return 1

    coll = _get_collection(args.db, args.collection, reset=args.rebuild)

    if not args.rebuild:
        existing = set()
        try:
            existing = set(coll.get(ids=ids)["ids"])  # type: ignore[index]
        except Exception:  # noqa: BLE001
            existing = set()
        if existing:
            pairs = [
                (i, d, m) for i, d, m in zip(ids, docs, metas) if i not in existing
            ]
            if not pairs:
                print(f"Nada nuevo que indexar. Documentos ya presentes: {len(existing)}")
                return 0
            ids, docs, metas = (
                [p[0] for p in pairs],
                [p[1] for p in pairs],
                [p[2] for p in pairs],
            )

    client = _get_openai_client()
    print(f"Generando embeddings para {len(docs)} descripciones con {args.model}...", file=sys.stderr)
    vectors = _embed_many(client, docs, args.model, args.batch)

    for chunk_ids, chunk_docs, chunk_metas, chunk_vecs in zip(
        _batched(ids, args.batch),
        _batched(docs, args.batch),
        _batched(metas, args.batch),
        _batched(vectors, args.batch),
    ):
        coll.add(
            ids=list(chunk_ids),
            documents=list(chunk_docs),
            metadatas=list(chunk_metas),
            embeddings=list(chunk_vecs),
        )

    print(f"OK. Indexados {len(ids)} documentos en '{args.collection}' ({args.db}).")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    coll = _get_collection(args.db, args.collection)
    client = _get_openai_client()
    vec = _embed_many(client, [args.query], args.model, args.batch)[0]

    where = None
    if args.fabricante:
        where = {"fabricante": args.fabricante}

    res = coll.query(query_embeddings=[vec], n_results=args.k, where=where)
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    if not ids:
        print("Sin resultados.")
        return 0

    for rank, (doc_id, doc, meta, dist) in enumerate(zip(ids, docs, metas, dists), start=1):
        similitud = 1.0 - float(dist)
        print(f"#{rank}  similitud={similitud:.4f}  id={doc_id}")
        print(f"   Nombre comercial: {meta.get('nombre_comercial', '')}")
        print(f"   Fabricante:       {meta.get('fabricante', '')}")
        print(f"   GMDN:             {meta.get('gmdn', '')}")
        desc = (doc or "").replace("\n", " ")
        if len(desc) > 220:
            desc = desc[:217] + "..."
        print(f"   Descripción:      {desc}")
        print()
    return 0


def _fetch_device_by_id(coll, device_id: str) -> Dict[str, str]:
    res = coll.get(ids=[device_id], include=["documents", "metadatas"])
    ids = res.get("ids") or []
    if not ids:
        raise KeyError(f"No se encontró el dispositivo con id={device_id}")
    meta = (res.get("metadatas") or [[{}]])[0] or {}
    doc = (res.get("documents") or [[""]])[0] or ""
    meta = dict(meta)
    meta["descripcion"] = meta.get("descripcion") or doc
    meta["id"] = device_id
    return meta


def _fetch_device_by_query(
    coll,
    openai_client: OpenAI,
    query: str,
    *,
    embed_model: str,
    batch: int,
) -> Dict[str, str]:
    vec = _embed_many(openai_client, [query], embed_model, batch)[0]
    res = coll.query(
        query_embeddings=[vec],
        n_results=1,
        include=["documents", "metadatas", "distances"],
    )
    ids = (res.get("ids") or [[]])[0]
    if not ids:
        raise LookupError(f"Sin resultados para la consulta: {query!r}")
    meta = dict((res.get("metadatas") or [[{}]])[0][0] or {})
    doc = ((res.get("documents") or [[""]])[0] or [""])[0] or ""
    meta["descripcion"] = meta.get("descripcion") or doc
    meta["id"] = ids[0]
    return meta


def _device_block(label: str, d: Dict[str, str]) -> str:
    return (
        f"[{label}]\n"
        f"- Nombre comercial: {d.get('nombre_comercial', '')}\n"
        f"- Fabricante:       {d.get('fabricante', '')}\n"
        f"- Código GMDN:      {d.get('gmdn', '')}\n"
        f"- Descripción técnica:\n{d.get('descripcion', '')}\n"
    )


COMPARE_SYSTEM_PROMPT = (
    "Eres un ingeniero clínico con experiencia en equivalencia técnica de dispositivos médicos. "
    "Analizas dos productos a partir de sus especificaciones y decides si son técnicamente "
    "equivalentes en material, calibre/medidas y uso clínico. Si algún dato no aparece en la "
    "descripción, indícalo explícitamente en vez de inventarlo. Responde SIEMPRE en español."
)


COMPARE_USER_TEMPLATE = (
    "Compara los siguientes dos dispositivos médicos:\n\n"
    "{a}\n"
    "{b}\n"
    "Pregunta: ¿Son estos dos productos técnicamente equivalentes en cuanto a material, "
    "calibre y uso clínico? Responde con un porcentaje de compatibilidad (0-100) y justifica "
    "las diferencias.\n\n"
    "Devuelve SOLO un JSON válido con esta forma exacta:\n"
    "{{\n"
    '  "porcentaje_compatibilidad": <entero 0-100>,\n'
    '  "resumen": "<1-2 frases>",\n'
    '  "material": {{"compatibles": <true|false|null>, "comentario": "<...>"}},\n'
    '  "calibre": {{"compatibles": <true|false|null>, "comentario": "<...>"}},\n'
    '  "uso_clinico": {{"compatibles": <true|false|null>, "comentario": "<...>"}},\n'
    '  "diferencias": ["<diferencia 1>", "<diferencia 2>"],\n'
    '  "datos_faltantes": ["<dato ausente 1>", "..."]\n'
    "}}\n"
    "Usa null cuando la descripción no aporte información suficiente para decidir ese aspecto."
)


def comparar_dispositivos(
    dispositivo_a: Dict[str, str],
    dispositivo_b: Dict[str, str],
    *,
    openai_client: Optional[OpenAI] = None,
    chat_model: str = DEFAULT_CHAT_MODEL,
) -> Dict[str, Any]:
    """
    Envía las especificaciones de dos dispositivos a GPT-4o y devuelve un dict con
    `porcentaje_compatibilidad` y una justificación estructurada.

    Cada dispositivo debe tener las claves: nombre_comercial, fabricante, gmdn, descripcion.
    """
    client = openai_client or _get_openai_client()
    user_msg = COMPARE_USER_TEMPLATE.format(
        a=_device_block("Dispositivo A", dispositivo_a),
        b=_device_block("Dispositivo B", dispositivo_b),
    )

    resp = client.chat.completions.create(
        model=chat_model,
        response_format={"type": "json_object"},
        temperature=0.1,
        messages=[
            {"role": "system", "content": COMPARE_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )
    raw = resp.choices[0].message.content or "{}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {"porcentaje_compatibilidad": None, "resumen": raw.strip(), "raw": raw}

    data["_dispositivo_a"] = {
        "id": dispositivo_a.get("id"),
        "nombre_comercial": dispositivo_a.get("nombre_comercial"),
        "fabricante": dispositivo_a.get("fabricante"),
        "gmdn": dispositivo_a.get("gmdn"),
    }
    data["_dispositivo_b"] = {
        "id": dispositivo_b.get("id"),
        "nombre_comercial": dispositivo_b.get("nombre_comercial"),
        "fabricante": dispositivo_b.get("fabricante"),
        "gmdn": dispositivo_b.get("gmdn"),
    }
    return data


def _print_compare_result(data: Dict[str, Any]) -> None:
    a = data.get("_dispositivo_a", {})
    b = data.get("_dispositivo_b", {})
    pct = data.get("porcentaje_compatibilidad")
    pct_str = f"{pct}%" if isinstance(pct, (int, float)) else "N/D"
    print(f"Compatibilidad estimada: {pct_str}")
    print(
        f"  A: {a.get('nombre_comercial', '')} — {a.get('fabricante', '')} (id={a.get('id')})"
    )
    print(
        f"  B: {b.get('nombre_comercial', '')} — {b.get('fabricante', '')} (id={b.get('id')})"
    )
    if data.get("resumen"):
        print(f"\nResumen: {data['resumen']}")

    for campo, titulo in (("material", "Material"), ("calibre", "Calibre"), ("uso_clinico", "Uso clínico")):
        sec = data.get(campo) or {}
        if sec:
            compat = sec.get("compatibles")
            print(f"\n{titulo}: compatibles={compat}")
            if sec.get("comentario"):
                print(f"  {sec['comentario']}")

    if data.get("diferencias"):
        print("\nDiferencias:")
        for d in data["diferencias"]:
            print(f"  - {d}")
    if data.get("datos_faltantes"):
        print("\nDatos faltantes:")
        for d in data["datos_faltantes"]:
            print(f"  - {d}")


def cmd_compare(args: argparse.Namespace) -> int:
    coll = _get_collection(args.db, args.collection)
    client = _get_openai_client()

    def resolve(id_arg: str, query_arg: str, etiqueta: str) -> Dict[str, str]:
        if id_arg:
            return _fetch_device_by_id(coll, id_arg)
        if query_arg:
            return _fetch_device_by_query(
                coll,
                client,
                query_arg,
                embed_model=args.model,
                batch=args.batch,
            )
        raise SystemExit(
            f"Debes indicar --id-{etiqueta} o --query-{etiqueta} para el dispositivo {etiqueta.upper()}."
        )

    dev_a = resolve(args.id_a, args.query_a, "a")
    dev_b = resolve(args.id_b, args.query_b, "b")

    data = comparar_dispositivos(
        dev_a,
        dev_b,
        openai_client=client,
        chat_model=args.chat_model,
    )

    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        _print_compare_result(data)
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    client = chromadb.PersistentClient(path=args.db)
    colls = [c.name for c in client.list_collections()]
    print(f"DB: {args.db}")
    print(f"Colecciones: {colls}")
    try:
        coll = client.get_collection(args.collection)
        print(f"'{args.collection}': {coll.count()} documentos")
    except Exception as err:  # noqa: BLE001
        print(f"No se pudo inspeccionar '{args.collection}': {err}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Embeddings OpenAI + ChromaDB para GUDID.")
    p.add_argument("--db", default=DEFAULT_DB, help=f"Ruta de ChromaDB (por defecto: {DEFAULT_DB})")
    p.add_argument("--collection", default=DEFAULT_COLLECTION, help="Nombre de la colección.")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Modelo de embeddings (por defecto: {DEFAULT_MODEL})")
    p.add_argument("--batch", type=int, default=DEFAULT_BATCH, help="Tamaño de batch para la API de embeddings.")

    sub = p.add_subparsers(dest="cmd", required=True)

    p_idx = sub.add_parser("index", help="Genera embeddings del CSV y los guarda en ChromaDB.")
    p_idx.add_argument("--csv", required=True, help="CSV de entrada (ej: gudid_filtrado.csv).")
    p_idx.add_argument(
        "--rebuild",
        action="store_true",
        help="Borra la colección antes de indexar.",
    )
    p_idx.set_defaults(func=cmd_index)

    p_s = sub.add_parser("search", help="Busca productos por significado.")
    p_s.add_argument("--query", required=True, help="Texto de búsqueda.")
    p_s.add_argument("--k", type=int, default=5, help="Número de resultados.")
    p_s.add_argument("--fabricante", default="", help="Filtro opcional por fabricante exacto.")
    p_s.set_defaults(func=cmd_search)

    p_c = sub.add_parser(
        "compare",
        help="Compara dos dispositivos con GPT-4o y devuelve un % de compatibilidad.",
    )
    p_c.add_argument("--id-a", default="", help="ID (hash) del dispositivo A en ChromaDB.")
    p_c.add_argument("--id-b", default="", help="ID (hash) del dispositivo B en ChromaDB.")
    p_c.add_argument(
        "--query-a",
        default="",
        help="Consulta semántica para elegir el dispositivo A (top-1).",
    )
    p_c.add_argument(
        "--query-b",
        default="",
        help="Consulta semántica para elegir el dispositivo B (top-1).",
    )
    p_c.add_argument(
        "--chat-model",
        default=DEFAULT_CHAT_MODEL,
        help=f"Modelo de chat para la comparación (por defecto: {DEFAULT_CHAT_MODEL}).",
    )
    p_c.add_argument("--json", action="store_true", help="Imprime el JSON completo devuelto por el modelo.")
    p_c.set_defaults(func=cmd_compare)

    p_i = sub.add_parser("info", help="Muestra información de la BD/colección.")
    p_i.set_defaults(func=cmd_info)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
