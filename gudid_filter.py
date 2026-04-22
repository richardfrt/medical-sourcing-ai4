#!/usr/bin/env python3
"""
Carga el dataset delimitado de AccessGUDID (GUDID) y filtra por una categoría
(por defecto buscando la cadena en productCodes y/o términos GMDN).

Requisitos: Python 3.9+ (solo biblioteca estándar).

Ejemplos:
  python gudid_filter.py --zip-path ./AccessGUDID_Delimited_Full_Release_20260401.zip --categoria Cardiovascular
  python gudid_filter.py --descargar --categoria Cardiovascular --salida cardiovascular.csv

Notas:
  - El ZIP completo pesa varios cientos de MB; la primera ejecución puede tardar.
  - Los archivos relevantes suelen ser data/device.txt, data/productCodes.txt y data/gmdnTerms.txt.
  - El código GMDN numérico puede o no estar presente según la versión del release; el script
    detecta columnas por cabecera (p. ej. gmdn_code) y, si no hay código, deja la celda vacía.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import re
import sys
import tempfile
import urllib.request
import zipfile
from collections import defaultdict
from typing import DefaultDict, Dict, Iterable, List, Optional, Sequence, Set, Tuple


DELIMITED_PAGE = "https://accessgudid.nlm.nih.gov/download/delimited"
DEFAULT_DEVICE_MEMBER = "data/device.txt"
DEFAULT_PRODUCT_MEMBER = "data/productCodes.txt"
DEFAULT_GMDN_MEMBER = "data/gmdnTerms.txt"


def _norm_name(name: str) -> str:
    x = name.strip().lower()
    x = x.replace("_", "")
    x = re.sub(r"\s+", "", x)
    return x


def _header_index_map(header: Sequence[str]) -> Dict[str, int]:
    return {_norm_name(h): i for i, h in enumerate(header) if h is not None}


def _pick_column(header_map: Dict[str, int], candidates: Sequence[str]) -> Optional[int]:
    for c in candidates:
        key = _norm_name(c)
        if key in header_map:
            return header_map[key]
    return None


def _discover_latest_delimited_zip_url() -> str:
    html = urllib.request.urlopen(DELIMITED_PAGE, timeout=120).read().decode("utf-8", "replace")
    z = re.findall(r'href="([^"]+AccessGUDID_Delimited_Full_Release_\d{8}\.zip)"', html, flags=re.I)
    if not z:
        raise RuntimeError(
            "No se pudo detectar automáticamente la URL del último ZIP delimitado. "
            "Visita https://accessgudid.nlm.nih.gov/download/delimited y pásala con --zip-url."
        )
    return z[0]


def _download(url: str, dest_path: str) -> None:
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    tmp = dest_path + ".part"
    try:
        with urllib.request.urlopen(url, timeout=600) as resp, open(tmp, "wb") as out:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        os.replace(tmp, dest_path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _open_zip_member(zf: zipfile.ZipFile, member: str) -> io.TextIOWrapper:
    try:
        raw = zf.open(member, "r")
    except KeyError as exc:
        raise FileNotFoundError(
            f"No se encontró '{member}' dentro del ZIP. "
            f"Miembros disponibles (primeros 40): {zf.namelist()[:40]}"
        ) from exc
    # zipfile en Python 3 devuelve bytes; decodificamos como UTF-8 con tolerancia.
    return io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline="")


def _read_pipe_csv_rows(path: str) -> Iterable[List[str]]:
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        yield from csv.reader(f, delimiter="|", quotechar='"')


def _read_pipe_csv_rows_from_zip(zf: zipfile.ZipFile, member: str) -> Iterable[List[str]]:
    with _open_zip_member(zf, member) as f:
        yield from csv.reader(f, delimiter="|", quotechar='"')


def _collect_matching_primary_dis(
    rows: Iterable[List[str]],
    header: List[str],
    needle: str,
    *,
    scan_cols: Sequence[str],
) -> Set[str]:
    hmap = _header_index_map(header)
    di_i = _pick_column(hmap, ("primary_di", "PRIMARY_DI", "PrimaryDI"))
    if di_i is None:
        raise RuntimeError(f"No se encontró la columna primary_di en auxiliar: cabecera={header[:40]}")

    idxs = []
    for c in scan_cols:
        j = _pick_column(hmap, (c,))
        if j is not None:
            idxs.append(j)
    if not idxs:
        raise RuntimeError(
            f"No se encontraron columnas para filtrar entre {scan_cols}. Cabecera={header[:80]}"
        )

    needle_l = needle.casefold()
    out: Set[str] = set()
    for row in rows:
        if di_i >= len(row):
            continue
        di = row[di_i].strip()
        if not di:
            continue
        blob = " ".join((row[i] for i in idxs if i < len(row))).casefold()
        if needle_l in blob:
            out.add(di)
    return out


def _uniq_join(values: Sequence[str], sep: str = " | ") -> str:
    seen = set()
    ordered: List[str] = []
    for v in values:
        if v not in seen:
            seen.add(v)
            ordered.append(v)
    return sep.join(ordered)


def _collect_gmdn_codes_for_dis(
    rows: Iterable[List[str]],
    header: List[str],
    dis: Set[str],
) -> Dict[str, str]:
    """Lee gmdnTerms y devuelve primary_di -> códigos GMDN unidos (solo para DIs en `dis`)."""
    if not dis:
        return {}

    hmap = _header_index_map(header)
    di_i = _pick_column(hmap, ("primary_di", "PRIMARY_DI", "PrimaryDI"))
    if di_i is None:
        raise RuntimeError(f"No se encontró primary_di en gmdnTerms. Cabecera={header[:80]}")

    code_i = _pick_column(
        hmap,
        (
            "gmdn_code",
            "gmdntermcode",
            "gmdn_term_code",
            "gmdnptcode",
            "gmdn_pt_code",
        ),
    )
    if code_i is None:
        return {di: "" for di in dis}

    bucket: DefaultDict[str, List[str]] = defaultdict(list)
    for row in rows:
        if di_i >= len(row):
            continue
        di = row[di_i].strip()
        if not di or di not in dis:
            continue
        if code_i < len(row):
            v = row[code_i].strip()
            if v:
                bucket[di].append(v)

    return {di: _uniq_join(bucket[di]) if bucket[di] else "" for di in dis}


def filtrar_gudid(
    *,
    zip_path: str,
    categoria: str,
    salida: str,
    device_member: str,
    product_member: str,
    gmdn_member: str,
    solo_producto: bool,
    solo_gmdn: bool,
) -> Tuple[int, int]:
    if not categoria.strip():
        raise ValueError("La categoría no puede estar vacía.")

    with zipfile.ZipFile(zip_path, "r") as zf:
        # --- Cabeceras ---
        prod_header = next(iter(_read_pipe_csv_rows_from_zip(zf, product_member)))
        gmdn_header = next(iter(_read_pipe_csv_rows_from_zip(zf, gmdn_member)))
        dev_header = next(iter(_read_pipe_csv_rows_from_zip(zf, device_member)))

        # --- Conjunto de DIs que coinciden con la categoría ---
        match_fields_product = ("product_code_name", "productCodeName", "PRODUCT_CODE_NAME")

        matched: Set[str] = set()
        if not solo_gmdn:
            rows_p = _read_pipe_csv_rows_from_zip(zf, product_member)
            next(rows_p, None)  # skip header
            matched |= _collect_matching_primary_dis(
                rows_p, prod_header, categoria, scan_cols=match_fields_product
            )

        if not solo_producto:
            rows_g = _read_pipe_csv_rows_from_zip(zf, gmdn_member)
            next(rows_g, None)
            matched |= _collect_matching_primary_dis(
                rows_g, gmdn_header, categoria, scan_cols=("gmdn_pt_name", "gmdn_pt_definition")
            )

        rows_g2 = _read_pipe_csv_rows_from_zip(zf, gmdn_member)
        next(rows_g2, None)
        gmdn_codes = _collect_gmdn_codes_for_dis(rows_g2, gmdn_header, matched)

        # --- Volcar devices ---
        hmap_d = _header_index_map(dev_header)
        di_i = _pick_column(hmap_d, ("primary_di", "PRIMARY_DI"))
        brand_i = _pick_column(hmap_d, ("brand_name", "brandName"))
        desc_i = _pick_column(hmap_d, ("device_description", "deviceDescription"))
        comp_i = _pick_column(hmap_d, ("company_name", "companyName"))
        if None in (di_i, brand_i, desc_i, comp_i):
            raise RuntimeError(
                "No se pudieron localizar columnas esperadas en device.txt. "
                f"Cabecera (primeros campos): {dev_header[:40]}"
            )

        os.makedirs(os.path.dirname(salida) or ".", exist_ok=True)
        total = 0
        kept = 0
        with open(salida, "w", newline="", encoding="utf-8") as out_f:
            w = csv.writer(out_f)
            w.writerow(
                [
                    "Nombre Comercial",
                    "Descripción Técnica",
                    "Fabricante",
                    "Código GMDN (Categoría Global)",
                ]
            )

            rows_d = _read_pipe_csv_rows_from_zip(zf, device_member)
            next(rows_d, None)
            for row in rows_d:
                total += 1
                if di_i >= len(row):
                    continue
                di = row[di_i].strip()
                if di not in matched:
                    continue
                kept += 1
                brand = row[brand_i] if brand_i < len(row) else ""
                desc = row[desc_i] if desc_i < len(row) else ""
                comp = row[comp_i] if comp_i < len(row) else ""
                code = gmdn_codes.get(di, "")
                w.writerow([brand, desc, comp, code])

        return total, kept


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Filtra GUDID (AccessGUDID delimitado) por categoría.")
    p.add_argument("--categoria", default="Cardiovascular", help="Texto a buscar (por defecto: Cardiovascular).")
    p.add_argument("--zip-path", help="Ruta local al ZIP AccessGUDID_Delimited_Full_Release_YYYYMMDD.zip")
    p.add_argument("--zip-url", help="URL directa del ZIP delimitado (si no se usa --zip-path).")
    p.add_argument(
        "--descargar",
        action="store_true",
        help="Descarga el último ZIP delimitado automáticamente (pesa varios cientos de MB).",
    )
    p.add_argument(
        "--cache-zip",
        default="",
        help="Ruta donde guardar / reutilizar el ZIP descargado (por defecto: carpeta temporal del sistema).",
    )
    p.add_argument("--salida", default="gudid_filtrado.csv", help="CSV de salida (UTF-8).")
    p.add_argument("--device-member", default=DEFAULT_DEVICE_MEMBER)
    p.add_argument("--product-member", default=DEFAULT_PRODUCT_MEMBER)
    p.add_argument("--gmdn-member", default=DEFAULT_GMDN_MEMBER)
    p.add_argument(
        "--solo-producto",
        action="store_true",
        help="Filtrar solo por productCodes.txt (ignora coincidencias en GMDN).",
    )
    p.add_argument(
        "--solo-gmdn",
        action="store_true",
        help="Filtrar solo por gmdnTerms.txt (ignora coincidencias en códigos de producto).",
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    if args.solo_producto and args.solo_gmdn:
        p.error("No puedes combinar --solo-producto y --solo-gmdn.")

    zip_path = args.zip_path
    if args.descargar and zip_path:
        p.error("Usa solo uno: --descargar o --zip-path.")

    if not zip_path:
        if args.zip_url:
            url = args.zip_url
        elif args.descargar:
            url = _discover_latest_delimited_zip_url()
        else:
            p.error(
                "Debes indicar --zip-path al ZIP local, o bien --descargar / --zip-url. "
                "Puedes obtener el enlace en: https://accessgudid.nlm.nih.gov/download/delimited"
            )

        if args.cache_zip:
            cache = args.cache_zip
        else:
            cache_dir = tempfile.mkdtemp(prefix="gudid_zip_")
            cache = os.path.join(cache_dir, os.path.basename(url))

        if not os.path.exists(cache):
            print(f"Descargando: {url}", file=sys.stderr)
            print(f"Destino: {cache}", file=sys.stderr)
            _download(url, cache)
        zip_path = cache

    assert zip_path is not None
    if not os.path.isfile(zip_path):
        raise FileNotFoundError(f"No existe el ZIP: {zip_path}")

    total, kept = filtrar_gudid(
        zip_path=zip_path,
        categoria=args.categoria,
        salida=args.salida,
        device_member=args.device_member,
        product_member=args.product_member,
        gmdn_member=args.gmdn_member,
        solo_producto=args.solo_producto,
        solo_gmdn=args.solo_gmdn,
    )
    print(f"Filas device leídas: {total}", file=sys.stderr)
    print(f"Filas exportadas: {kept}", file=sys.stderr)
    print(f"CSV escrito en: {os.path.abspath(args.salida)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
