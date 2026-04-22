#!/usr/bin/env python3
"""
Entity Resolution sobre la ChromaDB construida por gudid_embeddings.py.

Agrupa dispositivos médicos que representan el MISMO producto genérico pese a
tener nombres comerciales o fabricantes distintos.

Flujo:
    1) Descarga todos los embeddings + metadatos de la colección ChromaDB.
    2) Para cada dispositivo busca sus vecinos más cercanos (coseno) en la misma BD.
    3) Une pares con similitud >= --threshold (Union-Find).
    4) Opcionalmente exige que compartan código GMDN (--strict-gmdn).
    5) Opcionalmente verifica pares en zona gris con GPT-4o (--llm-verify).
    6) Exporta:
         - CSV con una fila por dispositivo y su group_id.
         - CSV resumen por grupo (tamaño, fabricantes, nombre canónico, GMDN).

Requisitos:
    pip install openai chromadb

Uso:
    python entity_resolution.py --out-devices devices_grouped.csv --out-groups groups_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from gudid_embeddings import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_COLLECTION,
    DEFAULT_DB,
    _get_collection,
    _get_openai_client,
    comparar_dispositivos,
)


# ---------- Union-Find ----------


class DSU:
    def __init__(self, items: Sequence[str]) -> None:
        self.parent: Dict[str, str] = {x: x for x in items}
        self.rank: Dict[str, int] = {x: 0 for x in items}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


# ---------- Utilidades ----------


def _gmdn_tokens(value: str) -> set:
    if not value:
        return set()
    return {t.strip() for t in value.split("|") if t.strip()}


def _gmdn_compatible(a: str, b: str) -> bool:
    """True si comparten algún código GMDN, o al menos uno es vacío (dato faltante)."""
    ta, tb = _gmdn_tokens(a), _gmdn_tokens(b)
    if not ta or not tb:
        return True
    return bool(ta & tb)


def _iter_batches(xs: Sequence, n: int) -> Iterable[Sequence]:
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


def _load_all(coll) -> Tuple[List[str], List[List[float]], List[Dict[str, Any]], List[str]]:
    """Descarga todos los vectores y metadatos de la colección."""
    total = coll.count()
    if total == 0:
        return [], [], [], []

    ids: List[str] = []
    embs: List[List[float]] = []
    metas: List[Dict[str, Any]] = []
    docs: List[str] = []

    offset = 0
    page = 1000
    while offset < total:
        part = coll.get(
            include=["embeddings", "metadatas", "documents"],
            limit=page,
            offset=offset,
        )
        p_ids = part.get("ids") or []
        p_embs = part.get("embeddings") or []
        p_metas = part.get("metadatas") or []
        p_docs = part.get("documents") or []
        ids.extend(p_ids)
        embs.extend(list(e) for e in p_embs)
        metas.extend(dict(m or {}) for m in p_metas)
        docs.extend(d or "" for d in p_docs)
        offset += len(p_ids)
        if not p_ids:
            break
    return ids, embs, metas, docs


# ---------- Núcleo ----------


def _find_candidate_pairs(
    coll,
    ids: Sequence[str],
    embs: Sequence[Sequence[float]],
    *,
    k_neighbors: int,
    threshold_pair: float,
    llm_band_low: Optional[float],
    batch: int,
) -> Tuple[List[Tuple[str, str, float]], List[Tuple[str, str, float]]]:
    """
    Devuelve (pares_fuertes, pares_borde) donde:
      - pares_fuertes: similitud >= threshold_pair
      - pares_borde: llm_band_low <= similitud < threshold_pair (si llm_band_low no es None)
    """
    id_to_idx = {i: k for k, i in enumerate(ids)}
    fuertes: Dict[Tuple[str, str], float] = {}
    borde: Dict[Tuple[str, str], float] = {}

    for chunk_ids, chunk_embs in zip(_iter_batches(ids, batch), _iter_batches(embs, batch)):
        res = coll.query(
            query_embeddings=list(chunk_embs),
            n_results=min(k_neighbors + 1, max(2, len(ids))),
            include=["distances"],
        )
        res_ids = res.get("ids") or []
        res_dists = res.get("distances") or []
        for src, neigh_ids, neigh_dists in zip(chunk_ids, res_ids, res_dists):
            for nid, nd in zip(neigh_ids, neigh_dists):
                if nid == src or nid not in id_to_idx:
                    continue
                sim = 1.0 - float(nd)
                key = (src, nid) if src < nid else (nid, src)
                if sim >= threshold_pair:
                    if sim > fuertes.get(key, -1.0):
                        fuertes[key] = sim
                elif llm_band_low is not None and sim >= llm_band_low:
                    if sim > borde.get(key, -1.0):
                        borde[key] = sim

    fuertes_list = sorted(((a, b, s) for (a, b), s in fuertes.items()), key=lambda x: -x[2])
    borde_list = sorted(((a, b, s) for (a, b), s in borde.items()), key=lambda x: -x[2])
    return fuertes_list, borde_list


def _build_device(idx: int, ids, metas, docs) -> Dict[str, str]:
    meta = metas[idx] or {}
    desc = meta.get("descripcion") or docs[idx] or ""
    return {
        "id": ids[idx],
        "nombre_comercial": str(meta.get("nombre_comercial", "") or ""),
        "fabricante": str(meta.get("fabricante", "") or ""),
        "gmdn": str(meta.get("gmdn", "") or ""),
        "descripcion": desc,
    }


def _canonical_info(group_indices: Sequence[int], metas: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    names = [str(metas[i].get("nombre_comercial", "") or "") for i in group_indices if metas[i]]
    companies = [str(metas[i].get("fabricante", "") or "") for i in group_indices if metas[i]]
    gmdns_flat: List[str] = []
    for i in group_indices:
        for t in _gmdn_tokens(str(metas[i].get("gmdn", "") or "")):
            gmdns_flat.append(t)

    name_counts = Counter(n for n in names if n)
    company_counts = Counter(c for c in companies if c)
    gmdn_counts = Counter(gmdns_flat)

    def top(counter: Counter) -> str:
        return counter.most_common(1)[0][0] if counter else ""

    return {
        "canonical_name": top(name_counts),
        "canonical_company": top(company_counts),
        "canonical_gmdn": top(gmdn_counts),
        "unique_brand_names": sorted({n for n in names if n}),
        "unique_companies": sorted({c for c in companies if c}),
        "unique_gmdns": sorted(gmdn_counts.keys()),
    }


def resolver_entidades(
    *,
    db_path: str,
    collection_name: str,
    out_devices: str,
    out_groups: str,
    k_neighbors: int = 10,
    threshold: float = 0.88,
    strict_gmdn: bool = False,
    llm_verify: bool = False,
    llm_band_low: float = 0.78,
    llm_threshold: int = 70,
    chat_model: str = DEFAULT_CHAT_MODEL,
    batch: int = 64,
    max_llm_pairs: int = 200,
) -> Tuple[int, int]:
    coll = _get_collection(db_path, collection_name)
    ids, embs, metas, docs = _load_all(coll)
    if not ids:
        print("La colección está vacía.", file=sys.stderr)
        return 0, 0
    print(f"Cargados {len(ids)} dispositivos desde ChromaDB.", file=sys.stderr)

    band_low: Optional[float] = llm_band_low if llm_verify else None
    fuertes, borde = _find_candidate_pairs(
        coll,
        ids,
        embs,
        k_neighbors=k_neighbors,
        threshold_pair=threshold,
        llm_band_low=band_low,
        batch=batch,
    )
    print(f"Pares candidatos fuertes (sim>={threshold}): {len(fuertes)}", file=sys.stderr)
    if llm_verify:
        print(f"Pares en banda gris [{band_low}, {threshold}): {len(borde)}", file=sys.stderr)

    dsu = DSU(ids)
    id_to_idx = {i: k for k, i in enumerate(ids)}

    def _try_union(a: str, b: str) -> bool:
        if strict_gmdn:
            ga = str(metas[id_to_idx[a]].get("gmdn", "") or "")
            gb = str(metas[id_to_idx[b]].get("gmdn", "") or "")
            if not _gmdn_compatible(ga, gb):
                return False
        dsu.union(a, b)
        return True

    for a, b, _sim in fuertes:
        _try_union(a, b)

    if llm_verify and borde:
        pairs_to_verify = borde[:max_llm_pairs]
        client = _get_openai_client()
        print(
            f"Verificando con {chat_model} {len(pairs_to_verify)} pares de borde "
            f"(llm_threshold={llm_threshold})...",
            file=sys.stderr,
        )
        for idx_pair, (a, b, sim) in enumerate(pairs_to_verify, start=1):
            if dsu.find(a) == dsu.find(b):
                continue
            dev_a = _build_device(id_to_idx[a], ids, metas, docs)
            dev_b = _build_device(id_to_idx[b], ids, metas, docs)
            try:
                res = comparar_dispositivos(dev_a, dev_b, openai_client=client, chat_model=chat_model)
            except Exception as err:  # noqa: BLE001
                print(f"  [warn] comparación falló para ({a},{b}): {err}", file=sys.stderr)
                time.sleep(0.5)
                continue
            pct = res.get("porcentaje_compatibilidad")
            if isinstance(pct, (int, float)) and pct >= llm_threshold:
                _try_union(a, b)
            if idx_pair % 25 == 0:
                print(f"  ... {idx_pair}/{len(pairs_to_verify)} pares verificados", file=sys.stderr)

    groups: Dict[str, List[int]] = defaultdict(list)
    for i, device_id in enumerate(ids):
        groups[dsu.find(device_id)].append(i)

    group_order = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    id_to_group: Dict[str, int] = {}
    group_info: List[Dict[str, Any]] = []

    for new_gid, (_root, members) in enumerate(group_order, start=1):
        info = _canonical_info(members, metas)
        info["group_id"] = new_gid
        info["size"] = len(members)
        group_info.append(info)
        for idx in members:
            id_to_group[ids[idx]] = new_gid

    os.makedirs(os.path.dirname(out_devices) or ".", exist_ok=True)
    with open(out_devices, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "group_id",
                "id",
                "nombre_comercial",
                "fabricante",
                "gmdn",
                "descripcion",
            ]
        )
        for i, device_id in enumerate(ids):
            m = metas[i] or {}
            w.writerow(
                [
                    id_to_group[device_id],
                    device_id,
                    m.get("nombre_comercial", ""),
                    m.get("fabricante", ""),
                    m.get("gmdn", ""),
                    (m.get("descripcion") or docs[i] or "").replace("\r", " ").replace("\n", " "),
                ]
            )

    with open(out_groups, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "group_id",
                "size",
                "canonical_name",
                "canonical_company",
                "canonical_gmdn",
                "unique_brand_names",
                "unique_companies",
                "unique_gmdns",
            ]
        )
        for g in group_info:
            w.writerow(
                [
                    g["group_id"],
                    g["size"],
                    g["canonical_name"],
                    g["canonical_company"],
                    g["canonical_gmdn"],
                    " | ".join(g["unique_brand_names"]),
                    " | ".join(g["unique_companies"]),
                    " | ".join(g["unique_gmdns"]),
                ]
            )

    grupos_multi = sum(1 for g in group_info if g["size"] > 1)
    print(
        f"OK. Dispositivos: {len(ids)}. Grupos: {len(group_info)}. "
        f"Grupos con ≥2 miembros: {grupos_multi}.",
        file=sys.stderr,
    )
    print(f"CSV dispositivos: {os.path.abspath(out_devices)}", file=sys.stderr)
    print(f"CSV grupos:        {os.path.abspath(out_groups)}", file=sys.stderr)
    return len(ids), len(group_info)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Entity Resolution sobre la ChromaDB del GUDID.")
    p.add_argument("--db", default=DEFAULT_DB)
    p.add_argument("--collection", default=DEFAULT_COLLECTION)
    p.add_argument("--k-neighbors", type=int, default=10, help="Vecinos por dispositivo para buscar pares.")
    p.add_argument("--threshold", type=float, default=0.88, help="Umbral de similitud para unir automáticamente.")
    p.add_argument(
        "--strict-gmdn",
        action="store_true",
        help="Solo une si comparten algún código GMDN (o uno está vacío).",
    )
    p.add_argument("--llm-verify", action="store_true", help="Usa GPT-4o para pares en zona gris.")
    p.add_argument("--llm-band-low", type=float, default=0.78, help="Límite inferior de la banda gris.")
    p.add_argument(
        "--llm-threshold",
        type=int,
        default=70,
        help="Porcentaje mínimo de equivalencia técnica para unir vía LLM.",
    )
    p.add_argument("--chat-model", default=DEFAULT_CHAT_MODEL)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--max-llm-pairs", type=int, default=200, help="Límite de pares a verificar con LLM.")
    p.add_argument("--out-devices", default="devices_grouped.csv")
    p.add_argument("--out-groups", default="groups_summary.csv")
    args = p.parse_args(list(argv) if argv is not None else None)

    resolver_entidades(
        db_path=args.db,
        collection_name=args.collection,
        out_devices=args.out_devices,
        out_groups=args.out_groups,
        k_neighbors=args.k_neighbors,
        threshold=args.threshold,
        strict_gmdn=args.strict_gmdn,
        llm_verify=args.llm_verify,
        llm_band_low=args.llm_band_low,
        llm_threshold=args.llm_threshold,
        chat_model=args.chat_model,
        batch=args.batch,
        max_llm_pairs=args.max_llm_pairs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
