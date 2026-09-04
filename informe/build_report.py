"""Genera el informe ejecutivo desde los resultados finales congelados."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether,
                               PageBreak, PageTemplate, Paragraph, Spacer,
                               Table, TableStyle)

ROOT = Path(__file__).parents[1]
OUT = ROOT / "informe.pdf"
SOURCE = ROOT / "informe/informe.md"

NAVY = colors.HexColor("#17324D")
BLUE = colors.HexColor("#2D6A8A")
TEAL = colors.HexColor("#4B8178")
PALE = colors.HexColor("#EAF1F5")
GREEN = colors.HexColor("#DDECDD")
RED = colors.HexColor("#F6E2DD")
GRAY = colors.HexColor("#5A6670")


def load_json(path: str):
    return json.loads((ROOT / path).read_text())


def money(value: float) -> str:
    return f"Q{value:,.0f}"


def page(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, letter[1] - 0.22 * inch, letter[0], 0.22 * inch, fill=1, stroke=0)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(GRAY)
    canvas.drawString(0.55 * inch, 0.32 * inch, "Banco del Altiplano | Monitoreo transaccional")
    canvas.drawRightString(letter[0] - 0.55 * inch, 0.32 * inch, f"Página {doc.page}")
    canvas.restoreState()


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", parent=base["Title"], fontName="Helvetica-Bold", fontSize=20,
                                leading=23, textColor=NAVY, alignment=TA_LEFT, spaceAfter=5),
        "subtitle": ParagraphStyle("subtitle", parent=base["Normal"], fontSize=9.5, leading=12,
                                   textColor=GRAY, spaceAfter=10),
        "h1": ParagraphStyle("h1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=14,
                             leading=17, textColor=NAVY, spaceBefore=3, spaceAfter=7),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=10.5,
                             leading=13, textColor=BLUE, spaceBefore=6, spaceAfter=4),
        "body": ParagraphStyle("body", parent=base["BodyText"], fontName="Helvetica", fontSize=9.1,
                               leading=12.2, textColor=colors.HexColor("#24313A"), spaceAfter=5),
        "small": ParagraphStyle("small", parent=base["BodyText"], fontSize=7.7, leading=9.7,
                                textColor=colors.HexColor("#303A42"), spaceAfter=3),
        "callout": ParagraphStyle("callout", parent=base["BodyText"], fontName="Helvetica-Bold",
                                  fontSize=10.3, leading=13.3, textColor=NAVY, spaceAfter=4),
        "matrix": ParagraphStyle("matrix", parent=base["BodyText"], fontSize=6.6, leading=8,
                                 textColor=colors.HexColor("#26343D")),
        "caption": ParagraphStyle("caption", parent=base["BodyText"], fontSize=7.2, leading=9,
                                  alignment=TA_CENTER, textColor=GRAY, spaceAfter=5),
    }


def p(text, style):
    return Paragraph(text, style)


def table(data, widths, st, font=7.5, header=True):
    header_style = ParagraphStyle("th", parent=st["small"], textColor=colors.white,
                                  fontName="Helvetica-Bold", fontSize=font)
    converted = [[p(str(cell), header_style if header and row_index == 0 else st["small"])
                  for cell in row] for row_index, row in enumerate(data)]
    result = Table(converted, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("GRID", (0, 0), (-1, -1), .35, colors.HexColor("#AAB7C0")),
                ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        commands += [("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                     ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
    for row in range(1 if header else 0, len(data)):
        if row % 2 == 0:
            commands.append(("BACKGROUND", (0, row), (-1, row), PALE))
    result.setStyle(TableStyle(commands))
    return result


def img(path: str, width: float, height: float):
    item = Image(str(ROOT / path), width=width, height=height)
    item.hAlign = "CENTER"
    return item


def build_markdown(r, e, f, matrix):
    models = r["models"]
    rows = "\n".join(f"| {m} | {x['validation_ap']:.4f} | {x['test_ap']:.4f} | {x['recall']:.3f} | {x['f1']:.3f} | {money(x['cost_gtq'])} |" for m, x in models.items())
    matrix_rows = "\n".join(f"| {x['evidence']} | {x['figure_or_table']} | {x['conclusion']} | {x['limitation']} |" for x in matrix)
    SOURCE.write_text(f"""# Monitoreo transaccional: ¿qué revela el orden de las operaciones?

**Audiencia:** Comité de Riesgos del Banco del Altiplano  
**Curso:** Deep Learning 2026 | **Equipo:** Proyecto 1

**Repositorio del proyecto:** [https://github.com/Nery2004/Arquitecturas-Deep-Learning](https://github.com/Nery2004/Arquitecturas-Deep-Learning)

## Resumen ejecutivo

Comparamos A (agregados), B (GRU secuencial) y C (híbrido). B sí utilizó el orden: su AP de VALIDATION cayó de {f['original_validation_ap']:.4f} a {f['permuted_mean_ap']:.4f} al permutar las mismas operaciones. Sin embargo, A obtuvo el mejor AP de TEST ({models['A']['test_ap']:.4f}) y el menor costo ({money(models['A']['cost_gtq'])}). El candidato es A, el ahorro frente a A es Q0 y recomendamos **CONSERVAR**.

## Datos y protocolo

Dataset sintético: {r['dataset']['transactions']:,} transacciones, {r['dataset']['cards']:,} tarjetas, fraude {r['dataset']['fraud_rate']:.4%}. Split temporal y preprocessing fit solo con TRAIN.

Cada ejemplo representa una transacción objetivo y hasta 12 operaciones anteriores. El historial usa solo eventos pasados; TRAIN precede VALIDATION y VALIDATION precede TEST. El modelo y sus thresholds quedaron congelados antes de abrir TEST.

## Comparación A vs B

A recibe la operación actual y 23 agregados causales. B recibe la operación actual y la historia ordenada mediante una GRU. En VALIDATION, A obtuvo {models['A']['validation_ap']:.4f} AP y B {models['B']['validation_ap']:.4f}. AP se usa porque el fraude es minoritario y concentra la evaluación en la clase fraudulenta.

## Valor del orden

Al permutar exactamente las mismas operaciones, B cayó de {f['original_validation_ap']:.4f} a {f['permuted_mean_ap']:.4f} AP en promedio; la caída absoluta fue {f['permutation_absolute_drop']:.4f}. Esto demuestra sensibilidad al orden, no superioridad frente a A. Al recortar la historia a tres eventos, AP fue {f['short_history_ap']:.4f}; la cantidad adicional de contexto aportó poco.

## Apuesta C

La hipótesis previa fue que secuencia y agregados describían aspectos complementarios. El control fue B. C alcanzó {models['C']['validation_ap']:.4f} AP, una mejora de {models['C']['validation_ap']-models['B']['validation_ap']:.4f}; la ablación de agregados cayó a 0.4295. Cumplió la parte predictiva, pero no el criterio operativo completo porque quedó debajo de A y tuvo mayor costo.

## Resultados finales

| Modelo | Validation AP | Test AP | Recall | F1 | Costo TEST |
|---|---:|---:|---:|---:|---:|
{rows}

## Decisión económica

Costo = FN × Q4,200 + FP × Q180. Threshold A: {models['A']['threshold']:.8f}. Equivalente mensual: {money(r['economics']['monthly_equivalent_candidate_gtq'])}. Escala sintética.

## Errores, límites y recomendación

A dejó pasar {models['A']['fn']} fraudes y bloqueó {models['A']['fp']} operaciones legítimas. Seis falsos negativos fueron amount_anomaly. Entre los falsos positivos hubo microcompras, viajes, rachas y una compra grande; no apareció un patrón único.

Recomendamos **CONSERVAR** A. Los datos son sintéticos, los costos son promedios simplificados, la cifra mensual conserva la escala simulada, la historia está limitada a 12 eventos y puede existir drift. Cambiaríamos la recomendación si B/C mostraran ahorro sostenido con datos reales o si A fallara en mecanismos críticos.

## Matriz de evidencias

| Evidencia | Figura o tabla donde aparece | Conclusión | Limitación |
|---|---|---|---|
{matrix_rows}
""", encoding="utf-8")


def build():
    r = load_json("artefactos/final_results.json")
    e = load_json("artefactos/final_evaluation.json")
    f = load_json("artefactos/model_b/falsification_metadata.json")
    with (ROOT / "evidence_matrix.csv").open(newline="", encoding="utf-8") as handle:
        matrix = list(csv.DictReader(handle))
    build_markdown(r, e, f, matrix)
    st = styles(); story = []; models = r["models"]
    doc = BaseDocTemplate(str(OUT), pagesize=letter, leftMargin=.55*inch, rightMargin=.55*inch,
                          topMargin=.45*inch, bottomMargin=.52*inch, title="Monitoreo transaccional: ¿qué revela el orden de las operaciones?",
                          author="Equipo del Proyecto 1")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main")
    doc.addPageTemplates([PageTemplate(id="report", frames=[frame], onPage=page)])

    # Página 1
    story += [Spacer(1, .05*inch), p("Monitoreo transaccional:<br/>¿qué revela el orden de las operaciones?", st["title"]),
              p("Evaluación del valor de modelos secuenciales para detección de fraude<br/><b>Comité de Riesgos del Banco del Altiplano</b> | Deep Learning 2026 | Equipo del Proyecto 1", st["subtitle"]),
              p("<b>Repositorio del proyecto:</b> <link href='https://github.com/Nery2004/Arquitecturas-Deep-Learning' color='#2D6A8A'>https://github.com/Nery2004/Arquitecturas-Deep-Learning</link>", st["small"]),
              p("Resumen ejecutivo", st["h1"]),
              p(f"El banco necesitaba saber si leer las operaciones en secuencia justificaba cambiar un sistema basado en promedios, conteos y señales agregadas. Comparamos tres alternativas: A, sin secuencia ordenada; B, una GRU que recibió el historial en orden; y C, que combinó ambas representaciones. B sí utilizó el orden: en VALIDATION, su Average Precision (AP) cayó de {f['original_validation_ap']:.4f} a {f['permuted_mean_ap']:.4f} cuando permutamos exactamente los mismos eventos. Sin embargo, esa señal no se convirtió en una ventaja frente a A. En TEST, A alcanzó AP {models['A']['test_ap']:.4f} y un costo estimado de {money(models['A']['cost_gtq'])}; B y C costaron {money(models['B']['cost_gtq'])} y {money(models['C']['cost_gtq'])}. El candidato final, definido antes de abrir TEST, fue A con threshold {models['A']['threshold']:.6f}. Su ahorro frente al propio A es Q0. Recomendamos <b>CONSERVAR</b>: mantener A y no incorporar secuencias todavía, aunque el hallazgo sobre el orden merece validarse con datos reales.", st["body"]),
              p("1. La decisión y los datos", st["h1"]),
              p("La pregunta no era si podíamos entrenar una red recurrente, sino si la secuencia añadía una señal útil, en qué fraudes y con qué impacto económico. El sistema actual se representa conceptualmente con A: operación actual más resúmenes causales del comportamiento de la tarjeta.", st["body"]),
              table([["Origen", "Transacciones", "Tarjetas", "Periodo", "Fraude"], ["Generador sintético propio", f"{r['dataset']['transactions']:,}", f"{r['dataset']['cards']:,}", "1 ene.-29 jun. 2025", f"{r['dataset']['fraud_rate']:.4%}"]], [1.55*inch,.95*inch,.75*inch,1.35*inch,.75*inch], st),
              p("El generador incluyó testing_cashout, channel_takeover y amount_anomaly, además de viajes, compras grandes, rachas y microcompras legítimas difíciles. Esta construcción permite controlar la señal, pero no equivale a comportamiento bancario real.", st["small"]), PageBreak()]

    # Página 2
    split_rows=[["Split","Periodo","Ejemplos","Fraudes"],*[ [k.upper(),f"{v['start'][:10]} a {v['end'][:10]}",f"{v['n']:,}",f"{v['frauds']:,}"] for k,v in r['splits'].items()]]
    story += [p("2. Protocolo: medir hacia adelante", st["h1"]),
              p("Cada ejemplo representa una transacción objetivo y hasta 12 operaciones anteriores de la misma tarjeta. TRAIN contiene el pasado más antiguo; VALIDATION sirvió para elegir modelos y thresholds; TEST quedó reservado para una única evaluación final.", st["body"]),
              table(split_rows,[1.0*inch,2.4*inch,1.0*inch,.9*inch],st),
              p("Controles de fuga de información", st["h2"]),
              p("El historial usa solo eventos anteriores al target. Scalers, imputación y vocabularios se ajustaron únicamente con TRAIN. Los tres modelos comparten targets y fronteras. El modelo candidato y los thresholds se congelaron antes de revisar TEST.", st["body"]),
              img("figures/eda_temporal_split.png", 5.85*inch, 2.35*inch), p("Figura 1. Partición cronológica común a los tres modelos.", st["caption"]),
              p("3. Qué conseguimos sin utilizar el orden", st["h1"]),
              p(f"A usa la operación actual y 23 resúmenes históricos; B usa una GRU con las operaciones en orden. Utilizamos AUC-PR/AP porque el fraude representa solo {r['dataset']['fraud_rate']:.2%} de las transacciones y esta métrica se concentra en la calidad de detección de la clase fraudulenta. En VALIDATION, A alcanzó {models['A']['validation_ap']:.4f} y B {models['B']['validation_ap']:.4f}. A fue claramente más competitivo.", st["body"]), PageBreak()]

    # Página 3
    story += [p("4. ¿El orden realmente agregó información?", st["h1"]),
              p("Tomamos exactamente las mismas operaciones de cada historial de VALIDATION y cambiamos únicamente el orden. No reentrenamos ni modificamos targets, longitudes o la operación actual.", st["body"]),
              table([["Condición","AP en VALIDATION","Cambio"], ["B original",f"{f['original_validation_ap']:.4f}","-"], ["B permutado (media)",f"{f['permuted_mean_ap']:.4f}",f"-{f['permutation_absolute_drop']:.4f}"], ["B: últimos 3 eventos",f"{f['short_history_ap']:.4f}",f"-{f['short_history_absolute_drop']:.4f}"]], [2.25*inch,1.55*inch,1.25*inch], st),
              img("figures/order_evidence_summary.png", 5.9*inch, 2.65*inch), p("Figura 2. A aporta la referencia sin secuencia; las falsificaciones corresponden a B en VALIDATION.", st["caption"]),
              p("La caída media tras permutar fue fuerte y consistente: 0.4363 AP, con desviación 0.0202 entre diez semillas. Esto aporta evidencia de que B estaba utilizando el orden. No prueba que B sea mejor que A ni que el mismo efecto aparecerá en datos reales.", st["callout"]),
              p("El recorte responde otra pregunta. Mantuvimos el orden, pero conservamos solo los tres eventos más recientes. AP pasó de 0.7207 a 0.7108. Para este checkpoint, el contexto entre el cuarto y el duodécimo evento aportó poco; orden y cantidad de historia no son lo mismo.", st["body"]), PageBreak()]

    # Página 4
    final_rows=[["Modelo","AP","Precision","Recall","F1","FN","FP","Costo"], *[[m,f"{x['test_ap']:.4f}",f"{x['precision']:.3f}",f"{x['recall']:.3f}",f"{x['f1']:.3f}",x['fn'],x['fp'],money(x['cost_gtq'])] for m,x in models.items()]]
    story += [p("5. Apuesta C: combinar secuencia y agregados", st["h1"]),
              p("La hipótesis escrita antes de entrenar fue que combinar la secuencia con los resúmenes históricos podía mejorar el resultado porque ambas representaciones describen aspectos distintos del comportamiento. El control fue B.", st["body"]),
              table([["Comparación en VALIDATION","AP","Diferencia"], ["B - control",f"{models['B']['validation_ap']:.4f}","-"], ["C - híbrido",f"{models['C']['validation_ap']:.4f}",f"+{models['C']['validation_ap']-models['B']['validation_ap']:.4f}"], ["C con agregados neutralizados",f"{load_json('artefactos/model_c/model_c_metadata.json')['ablation_aggregates_neutral_ap']:.4f}",f"-{load_json('artefactos/model_c/model_c_metadata.json')['ablation_difference']:.4f}"]], [2.9*inch,1.0*inch,1.0*inch],st),
              p("C superó a B y la ablación confirmó que la rama agregada sí aportaba. La hipótesis cumplió su parte predictiva, pero no el criterio operativo completo: C quedó debajo de A y tuvo mayor costo final. No premiamos la complejidad por sí misma.", st["callout"]),
              p("Resultados finales en TEST", st["h1"]),
              p("El modelo y cada threshold se definieron antes de revisar TEST. AP resume la calidad del ranking; precision indica qué proporción de los bloqueos era fraude; recall, qué proporción del fraude fue detectada; F1 resume el equilibrio entre ambas.", st["small"]),
              table(final_rows,[.52*inch,.57*inch,.72*inch,.62*inch,.52*inch,.38*inch,.38*inch,.72*inch],st,font=7),
              img("figures/test_pr_curves_abc.png", 4.95*inch, 2.35*inch), p("Figura 3. Curvas Precision-Recall en TEST, usadas solo como evidencia descriptiva final.", st["caption"]), PageBreak()]

    # Página 5
    story += [p("6. Umbral y valor económico", st["h1"]),
              p("Para traducir el score en una decisión usamos la aproximación acordada: cada fraude no detectado cuesta Q4,200 y cada operación legítima bloqueada cuesta Q180.", st["body"]),
              p("Costo = FN × Q4,200 + FP × Q180", ParagraphStyle("formula",parent=st["callout"],alignment=TA_CENTER,fontSize=14,leading=18,backColor=PALE,borderPadding=8)),
              p(f"Cada modelo recibió su propio threshold, elegido exclusivamente en VALIDATION. A quedó congelado en {models['A']['threshold']:.8f}. En TEST dejó pasar {models['A']['fn']} fraudes y bloqueó {models['A']['fp']} operaciones legítimas.", st["body"]),
              table([["Referencia","Costo TEST","Diferencia frente a A"], ["Never Block",money(e['test_baselines']['never_block_cost']),money(e['test_baselines']['never_block_cost']-models['A']['cost_gtq'])], ["A - candidato",money(models['A']['cost_gtq']),"Q0"], ["B - secuencial",money(models['B']['cost_gtq']),f"+{money(models['B']['cost_gtq']-models['A']['cost_gtq'])}"], ["C - híbrido",money(models['C']['cost_gtq']),f"+{money(models['C']['cost_gtq']-models['A']['cost_gtq'])}"]], [2.1*inch,1.3*inch,1.7*inch],st),
              img("figures/test_economic_cost_abc.png", 5.25*inch, 2.55*inch), p("Figura 4. Costo con thresholds congelados en VALIDATION.", st["caption"]),
              p(f"La comparación principal es contra A, porque representa lo que puede lograrse sin secuencia. El candidato final también es A: ahorro frente a A = Q0. TEST cubrió {r['economics']['test_days']:.2f} días; normalizado a 30.44 días, su costo equivalente es {money(r['economics']['monthly_equivalent_candidate_gtq'])} al mes. Esta cifra pertenece a la escala sintética y no debe extrapolarse a 1.4 millones de tarjetas.", st["callout"]), PageBreak()]

    # Página 6
    mech=e['candidate_mechanisms']
    story += [p("7. Errores, límites y recomendación", st["h1"]),
              table([["Mecanismo","Fraudes","Detectados","No detectados","Recall"], *[[k,v['n'],v['detected'],v['missed'],f"{v['recall']:.1%}"] for k,v in mech.items()]], [1.65*inch,.75*inch,.85*inch,.95*inch,.75*inch],st),
              img("figures/test_recall_by_mechanism_candidate.png", 5.5*inch, 2.3*inch), p("Figura 5. Recall del candidato A por mecanismo en TEST.", st["caption"]),
              p("Qué se equivocó", st["h2"]),
              p("Seis de los once fraudes no detectados fueron amount_anomaly; todos los casos representativos tenían 12 eventos de historia, por lo que no se explican por historial corto. Entre los falsos positivos hubo microcompras, viajes, rachas de compras y una compra legítima grande. No encontramos un patrón único para todas las falsas alarmas.", st["body"]),
              p("Recomendación: CONSERVAR", st["h1"]),
              p("Mantendríamos A. El orden contiene señal y C confirmó complementariedad, pero ni B ni C mejoraron AP o costo frente al modelo agregado. Antes de invertir en una solución secuencial, pediríamos una réplica con datos bancarios reales y costos operativos validados.", st["callout"]),
              p("Límites y condiciones para cambiar la decisión", st["h2"]),
              p("1) Los datos y mecanismos son sintéticos. 2) Q4,200/Q180 son costos medios simplificados. 3) La escala mensual no representa el universo real del banco. 4) La ventana de 12 eventos limita el contexto. 5) Puede existir drift. Cambiaríamos la recomendación si B/C mostraran ahorro sostenido en datos reales, si A perdiera recall en mecanismos críticos, si los falsos positivos fueran inaceptables o si aparecieran patrones donde la secuencia ofreciera una ventaja operativa verificable.", st["small"]), PageBreak()]

    # Página 7
    matrix_data=[["Evidencia","Figura o tabla donde aparece","Conclusión","Limitación"]]
    labels=["Integridad de datos","Comparación común A vs B","Valor del orden","Apuesta del equipo","Decisión económica","Recomendación y límites"]
    for label,row in zip(labels,matrix):
        matrix_data.append([label,row['figure_or_table'],row['conclusion'],row['limitation']])
    story += [p("8. Matriz de evidencias", st["h1"]),
              p("Esta matriz conecta cada afirmación central con una evidencia reproducible del repositorio. Las falsificaciones pertenecen a VALIDATION; la tabla final y los costos pertenecen a TEST.", st["body"]),
              Table([[p(str(c), ParagraphStyle("mhead",parent=st['matrix'],fontName='Helvetica-Bold',textColor=colors.white)) for c in matrix_data[0]]] +
                    [[p(str(c), st['matrix']) for c in row] for row in matrix_data[1:]],
                    colWidths=[1.12*inch,1.55*inch,1.83*inch,1.45*inch], repeatRows=1,
                    style=TableStyle([('BACKGROUND',(0,0),(-1,0),NAVY),('GRID',(0,0),(-1,-1),.35,colors.HexColor('#AAB7C0')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,PALE])])),
              Spacer(1,.18*inch), p("Decisión para el Banco del Altiplano", st["h1"]),
              p("Con la evidencia disponible, conservaríamos el enfoque agregado A. No descartamos el valor científico del orden: la permutación mostró que existe. Lo que no observamos fue valor económico incremental suficiente para justificar una arquitectura secuencial. La siguiente decisión debe basarse en una validación controlada con datos reales, no en mayor complejidad técnica.", st["callout"]),
              p("Fuentes internas: artefactos/final_results.json, artefactos/final_evaluation.json, artefactos/final_decision_config.json, metadata A/B/C, falsification_metadata.json, evidence_matrix.csv y audit_report.md.", st["small"])]

    doc.build(story)
    return OUT


if __name__ == "__main__":
    print(build())
