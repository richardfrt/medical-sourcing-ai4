#!/usr/bin/env python3
"""
Genera un informe PDF de sustitución de un dispositivo médico (A) por otro (B):

    "Si sustituye el Producto A por el Producto B en su inventario anual,
     su hospital ahorraría X euros."

Incluye:
  - Tabla comparativa de especificaciones lado a lado (A vs B).
  - Análisis de equivalencia técnica (material, calibre, uso clínico) vía GPT-4o.
  - Firma del Jefe de Servicio Médico para aprobación.

Requisitos:
    pip install reportlab openai chromadb

Ejemplos:
    python generar_informe_pdf.py --id-a <hash_a> --id-b <hash_b> --unidades 1200 --hospital "Hospital Ejemplo"
    python generar_informe_pdf.py --query-a "stent coronario everolimus" --query-b "stent coronario zotarolimus" --unidades 800

Si no se indican precios, se simulan de forma determinista a partir del id del
dispositivo (reproducibles), de la misma forma que en streamlit_app.py.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from datetime import datetime
from typing import Any, Dict, Optional, Sequence

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from gudid_embeddings import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_COLLECTION,
    DEFAULT_DB,
    DEFAULT_MODEL,
    _fetch_device_by_id,
    _fetch_device_by_query,
    _get_collection,
    _get_openai_client,
    comparar_dispositivos,
)


# ---------- Utilidades ----------


def precio_simulado(device_id: str, minimo: float = 40.0, maximo: float = 1800.0) -> float:
    """Precio reproducible a partir del id del dispositivo (solo ilustrativo)."""
    h = hashlib.sha256(device_id.encode("utf-8")).digest()
    n = int.from_bytes(h[:4], "big") / 2**32
    return round(minimo + n * (maximo - minimo), 2)


def fmt_eur(value: float) -> str:
    try:
        s = f"{value:,.2f}"
    except (TypeError, ValueError):
        return str(value)
    # es-ES: miles con punto, decimales con coma
    return s.replace(",", "X").replace(".", ",").replace("X", ".") + " €"


def fmt_pct(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.0f} %"
    return "N/D"


def safe_text(x: Any) -> str:
    if x is None:
        return "—"
    s = str(x).strip()
    return s if s else "—"


# ---------- Estilos ----------


def _build_styles() -> Dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    styles: Dict[str, ParagraphStyle] = {
        "title": ParagraphStyle(
            "TitleStyle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            textColor=colors.HexColor("#0f172a"),
            spaceAfter=4,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=10.5,
            leading=14,
            textColor=colors.HexColor("#475569"),
            spaceAfter=6,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=17,
            textColor=colors.HexColor("#0b1220"),
            spaceBefore=10,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#0f172a"),
        ),
        "bodySmall": ParagraphStyle(
            "BodySmall",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#334155"),
        ),
        "keyFinding": ParagraphStyle(
            "KeyFinding",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=16,
            textColor=colors.HexColor("#0b1220"),
        ),
        "muted": ParagraphStyle(
            "Muted",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11,
            textColor=colors.HexColor("#64748b"),
        ),
    }
    return styles


# ---------- Bloques ----------


def _tabla_comparativa(dev_a: Dict[str, str], dev_b: Dict[str, str], styles) -> Table:
    filas = [
        ["Campo", "Producto A (actual)", "Producto B (alternativa)"],
        ["Nombre comercial", safe_text(dev_a.get("nombre_comercial")), safe_text(dev_b.get("nombre_comercial"))],
        ["Fabricante", safe_text(dev_a.get("fabricante")), safe_text(dev_b.get("fabricante"))],
        ["Código GMDN", safe_text(dev_a.get("gmdn")), safe_text(dev_b.get("gmdn"))],
        ["ID GUDID (DI)", safe_text(dev_a.get("id")), safe_text(dev_b.get("id"))],
        [
            "Descripción técnica",
            Paragraph(safe_text(dev_a.get("descripcion")), styles["bodySmall"]),
            Paragraph(safe_text(dev_b.get("descripcion")), styles["bodySmall"]),
        ],
    ]
    t = Table(filas, colWidths=[38 * mm, 65 * mm, 65 * mm], repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 10),
                ("BACKGROUND", (0, 1), (0, -1), colors.HexColor("#f1f5f9")),
                ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (0, 0), (-1, 0), "LEFT"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
                ("FONTSIZE", (0, 1), (-1, -1), 9.5),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return t


def _tabla_economica(
    unidades: int,
    precio_a: float,
    precio_b: float,
) -> Table:
    total_a = unidades * precio_a
    total_b = unidades * precio_b
    ahorro_unit = precio_a - precio_b
    ahorro_anual = total_a - total_b
    ahorro_pct = (ahorro_anual / total_a * 100.0) if total_a > 0 else 0.0

    filas = [
        ["Concepto", "Valor"],
        ["Unidades anuales", f"{unidades:,}".replace(",", ".")],
        ["Precio unitario Producto A", fmt_eur(precio_a)],
        ["Precio unitario Producto B", fmt_eur(precio_b)],
        ["Coste anual actual (A)", fmt_eur(total_a)],
        ["Coste anual alternativa (B)", fmt_eur(total_b)],
        ["Ahorro por unidad", fmt_eur(ahorro_unit)],
        ["Ahorro anual estimado", fmt_eur(ahorro_anual)],
        ["Ahorro anual (%)", f"{ahorro_pct:.1f} %".replace(".", ",")],
    ]
    t = Table(filas, colWidths=[80 * mm, 88 * mm])
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 10),
        ("BACKGROUND", (0, 1), (0, -1), colors.HexColor("#f8fafc")),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("ALIGN", (1, 1), (1, -1), "RIGHT"),
        ("FONTNAME", (0, -2), (-1, -2), "Helvetica-Bold"),
        ("BACKGROUND", (0, -2), (-1, -2), colors.HexColor("#ecfeff")),
        ("TEXTCOLOR", (0, -2), (-1, -2), colors.HexColor("#0c4a6e")),
    ]
    t.setStyle(TableStyle(style_cmds))
    return t, ahorro_anual, ahorro_pct


def _tabla_equivalencia(resultado_cmp: Dict[str, Any], styles) -> Table:
    pct = resultado_cmp.get("porcentaje_compatibilidad")
    resumen = resultado_cmp.get("resumen") or ""

    def cell(sec: Dict[str, Any]) -> Any:
        if not sec:
            return Paragraph("—", styles["bodySmall"])
        compat = sec.get("compatibles")
        comentario = sec.get("comentario") or "—"
        etiqueta = (
            "Sí" if compat is True else ("No" if compat is False else "Información insuficiente")
        )
        return Paragraph(f"<b>{etiqueta}.</b> {comentario}", styles["bodySmall"])

    filas = [
        ["Dimensión", "Evaluación"],
        ["Compatibilidad global", Paragraph(f"<b>{fmt_pct(pct)}</b> — {safe_text(resumen)}", styles["body"])],
        ["Material", cell(resultado_cmp.get("material") or {})],
        ["Calibre / medidas", cell(resultado_cmp.get("calibre") or {})],
        ["Uso clínico", cell(resultado_cmp.get("uso_clinico") or {})],
    ]

    diferencias = resultado_cmp.get("diferencias") or []
    if diferencias:
        items_html = "".join(f"&bull; {safe_text(d)}<br/>" for d in diferencias[:8])
        filas.append(["Diferencias clave", Paragraph(items_html, styles["bodySmall"])])

    faltantes = resultado_cmp.get("datos_faltantes") or []
    if faltantes:
        items_html = "".join(f"&bull; {safe_text(d)}<br/>" for d in faltantes[:8])
        filas.append(["Datos no informados", Paragraph(items_html, styles["bodySmall"])])

    t = Table(filas, colWidths=[42 * mm, 126 * mm], repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 10),
                ("BACKGROUND", (0, 1), (0, -1), colors.HexColor("#f1f5f9")),
                ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
                ("FONTSIZE", (0, 1), (0, -1), 9.5),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return t


def _bloque_ahorro(
    hospital: str,
    dev_a: Dict[str, str],
    dev_b: Dict[str, str],
    unidades: int,
    ahorro_anual: float,
    styles,
) -> Table:
    nombre_a = safe_text(dev_a.get("nombre_comercial"))
    nombre_b = safe_text(dev_b.get("nombre_comercial"))
    texto = (
        f"Si sustituye el <b>{nombre_a}</b> por el <b>{nombre_b}</b> en su inventario anual de "
        f"<b>{unidades:,}</b>".replace(",", ".")
        + " unidades, "
        + f"<b>{safe_text(hospital)}</b> ahorraría "
        + f"<b>{fmt_eur(ahorro_anual)}</b>."
    )
    p = Paragraph(texto, styles["keyFinding"])
    t = Table([[p]], colWidths=[170 * mm])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#ecfeff")),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#06b6d4")),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    return t


def _bloque_firma(styles) -> Table:
    firma = Paragraph(
        "<b>Aprobación · Jefe de Servicio Médico</b><br/>"
        "Nombre y apellidos: ____________________________________________<br/><br/>"
        "Fecha: ______________________&nbsp;&nbsp;&nbsp;Firma y sello: ____________________",
        styles["body"],
    )
    t = Table([[firma]], colWidths=[170 * mm])
    t.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#94a3b8")),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (0, 0), (-1, -1), 14),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
            ]
        )
    )
    return t


# ---------- PDF ----------


def generar_informe_pdf(
    *,
    dev_a: Dict[str, str],
    dev_b: Dict[str, str],
    precio_a: float,
    precio_b: float,
    unidades_anuales: int,
    hospital: str,
    resultado_cmp: Optional[Dict[str, Any]],
    out_path: str,
) -> str:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    styles = _build_styles()

    doc = SimpleDocTemplate(
        out_path,
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="Informe de sustitución de dispositivo médico",
        author="Análisis GUDID",
    )

    story = []
    story.append(Paragraph("Informe de sustitución de dispositivo médico", styles["title"]))
    subtitulo = (
        f"{safe_text(hospital)} · Generado el "
        + datetime.now().strftime("%d/%m/%Y %H:%M")
    )
    story.append(Paragraph(subtitulo, styles["subtitle"]))
    story.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#cbd5e1")))
    story.append(Spacer(1, 8))

    tabla_econ, ahorro_anual, ahorro_pct = _tabla_economica(unidades_anuales, precio_a, precio_b)
    story.append(_bloque_ahorro(hospital, dev_a, dev_b, unidades_anuales, ahorro_anual, styles))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Comparativa técnica (lado a lado)", styles["h2"]))
    story.append(_tabla_comparativa(dev_a, dev_b, styles))
    story.append(Spacer(1, 10))

    if resultado_cmp is not None:
        story.append(Paragraph("Análisis de equivalencia técnica (GPT-4o)", styles["h2"]))
        story.append(_tabla_equivalencia(resultado_cmp, styles))
        story.append(Spacer(1, 10))

    story.append(Paragraph("Impacto económico anual", styles["h2"]))
    story.append(tabla_econ)
    story.append(Spacer(1, 12))

    story.append(KeepTogether(_bloque_firma(styles)))
    story.append(Spacer(1, 6))
    story.append(
        Paragraph(
            "Datos de dispositivos: AccessGUDID (FDA). Análisis técnico generado por modelos "
            "de lenguaje; requiere validación clínica antes de su aplicación.",
            styles["muted"],
        )
    )

    doc.build(story)
    print(
        f"Informe generado en {os.path.abspath(out_path)} · Ahorro anual {fmt_eur(ahorro_anual)} ({ahorro_pct:.1f}%)",
        file=sys.stderr,
    )
    return out_path


# ---------- CLI ----------


def _resolver_dispositivo(
    coll,
    client,
    *,
    id_arg: str,
    query_arg: str,
    embed_model: str,
    etiqueta: str,
) -> Dict[str, str]:
    if id_arg:
        return _fetch_device_by_id(coll, id_arg)
    if query_arg:
        return _fetch_device_by_query(
            coll, client, query_arg, embed_model=embed_model, batch=1
        )
    raise SystemExit(f"Debes indicar --id-{etiqueta} o --query-{etiqueta}.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Genera PDF de sustitución A→B con ahorro y comparativa técnica.")
    p.add_argument("--db", default=DEFAULT_DB)
    p.add_argument("--collection", default=DEFAULT_COLLECTION)
    p.add_argument("--model", default=DEFAULT_MODEL, help="Modelo de embeddings (para --query-*).")
    p.add_argument("--chat-model", default=DEFAULT_CHAT_MODEL, help="Modelo de chat para comparación.")
    p.add_argument("--id-a", default="")
    p.add_argument("--id-b", default="")
    p.add_argument("--query-a", default="")
    p.add_argument("--query-b", default="")
    p.add_argument("--unidades", type=int, required=True, help="Unidades anuales del inventario.")
    p.add_argument("--precio-a", type=float, default=None, help="Precio unitario A (€). Si se omite, se simula.")
    p.add_argument("--precio-b", type=float, default=None, help="Precio unitario B (€). Si se omite, se simula.")
    p.add_argument("--hospital", default="Hospital", help="Nombre del hospital o servicio.")
    p.add_argument("--out", default="informe_sustitucion.pdf", help="Ruta del PDF de salida.")
    p.add_argument("--no-llm", action="store_true", help="Omite la comparación con GPT-4o.")
    args = p.parse_args(list(argv) if argv is not None else None)

    if args.unidades <= 0:
        p.error("--unidades debe ser > 0.")

    coll = _get_collection(args.db, args.collection)
    client = _get_openai_client() if not args.no_llm or args.query_a or args.query_b else None

    dev_a = _resolver_dispositivo(
        coll, client, id_arg=args.id_a, query_arg=args.query_a,
        embed_model=args.model, etiqueta="a",
    )
    dev_b = _resolver_dispositivo(
        coll, client, id_arg=args.id_b, query_arg=args.query_b,
        embed_model=args.model, etiqueta="b",
    )

    precio_a = float(args.precio_a) if args.precio_a is not None else precio_simulado(dev_a["id"])
    precio_b = float(args.precio_b) if args.precio_b is not None else precio_simulado(dev_b["id"])

    resultado_cmp: Optional[Dict[str, Any]] = None
    if not args.no_llm:
        if client is None:
            client = _get_openai_client()
        resultado_cmp = comparar_dispositivos(
            dev_a, dev_b, openai_client=client, chat_model=args.chat_model
        )

    generar_informe_pdf(
        dev_a=dev_a,
        dev_b=dev_b,
        precio_a=precio_a,
        precio_b=precio_b,
        unidades_anuales=args.unidades,
        hospital=args.hospital,
        resultado_cmp=resultado_cmp,
        out_path=args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
