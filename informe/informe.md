# Monitoreo transaccional: ¿qué revela el orden de las operaciones?

**Audiencia:** Comité de Riesgos del Banco del Altiplano  
**Curso:** Deep Learning 2026 | **Equipo:** Proyecto 1

## Resumen ejecutivo

Comparamos A (agregados), B (GRU secuencial) y C (híbrido). B sí utilizó el orden: su AP de VALIDATION cayó de 0.7207 a 0.2843 al permutar las mismas operaciones. Sin embargo, A obtuvo el mejor AP de TEST (0.9191) y el menor costo (Q83,820). El candidato es A, el ahorro frente a A es Q0 y recomendamos **CONSERVAR**.

## Datos y protocolo

Dataset sintético: 95,767 transacciones, 2,800 tarjetas, fraude 1.6446%. Split temporal y preprocessing fit solo con TRAIN.

Cada ejemplo representa una transacción objetivo y hasta 12 operaciones anteriores. El historial usa solo eventos pasados; TRAIN precede VALIDATION y VALIDATION precede TEST. El modelo y sus thresholds quedaron congelados antes de abrir TEST.

## Comparación A vs B

A recibe la operación actual y 23 agregados causales. B recibe la operación actual y la historia ordenada mediante una GRU. En VALIDATION, A obtuvo 0.9100 AP y B 0.7207. AP se usa porque el fraude es minoritario y concentra la evaluación en la clase fraudulenta.

## Valor del orden

Al permutar exactamente las mismas operaciones, B cayó de 0.7207 a 0.2843 AP en promedio; la caída absoluta fue 0.4363. Esto demuestra sensibilidad al orden, no superioridad frente a A. Al recortar la historia a tres eventos, AP fue 0.7108; la cantidad adicional de contexto aportó poco.

## Apuesta C

La hipótesis previa fue que secuencia y agregados describían aspectos complementarios. El control fue B. C alcanzó 0.8152 AP, una mejora de 0.0945; la ablación de agregados cayó a 0.4295. Cumplió la parte predictiva, pero no el criterio operativo completo porque quedó debajo de A y tuvo mayor costo.

## Resultados finales

| Modelo | Validation AP | Test AP | Recall | F1 | Costo TEST |
|---|---:|---:|---:|---:|---:|
| A | 0.9100 | 0.9191 | 0.950 | 0.657 | Q83,820 |
| B | 0.7207 | 0.7150 | 0.878 | 0.450 | Q194,400 |
| C | 0.8152 | 0.8248 | 0.905 | 0.533 | Q147,780 |

## Decisión económica

Costo = FN × Q4,200 + FP × Q180. Threshold A: 0.05548898. Equivalente mensual: Q94,927. Escala sintética.

## Errores, límites y recomendación

A dejó pasar 11 fraudes y bloqueó 209 operaciones legítimas. Seis falsos negativos fueron amount_anomaly. Entre los falsos positivos hubo microcompras, viajes, rachas y una compra grande; no apareció un patrón único.

Recomendamos **CONSERVAR** A. Los datos son sintéticos, los costos son promedios simplificados, la cifra mensual conserva la escala simulada, la historia está limitada a 12 eventos y puede existir drift. Cambiaríamos la recomendación si B/C mostraran ahorro sostenido con datos reales o si A fallara en mecanismos críticos.

## Matriz de evidencias

| Evidencia | Figura o tabla donde aparece | Conclusión | Limitación |
|---|---|---|---|
| Integridad de datos | figures/eda_temporal_split.png; artefactos/final_results.json#splits | TRAIN precede VALIDATION y VALIDATION precede TEST; preprocessing fue fit solo con TRAIN. | La integridad se comprobó sobre datos sintéticos y reglas del generador. |
| Comparación A vs B | proyecto1_apellidos.ipynb §7–8; artefactos/final_results.json#models | A superó a B en Validation AP: 0.910021 vs 0.720674. | La diferencia mezcla representaciones y arquitecturas; no aísla por sí sola el orden. |
| Valor del orden | figures/order_permutation_validation.png; experiments/falsification_results.csv | Permutar los mismos eventos redujo AP de B de 0.720674 a 0.284325 ± 0.020168. | Demuestra sensibilidad al orden en B, no superioridad operativa frente a A. |
| Apuesta C | figures/model_abc_validation_comparison.png; artefactos/model_c/model_c_metadata.json | C mejoró B a 0.815180 AP y la ablación de agregados cayó a 0.429477. | C quedó debajo de A y tuvo mayor costo final. |
| Decisión económica | figures/economic_cost_vs_threshold_validation.png; figures/test_economic_cost_abc.png | A fue prefijado con threshold 0.055489 y tuvo el menor costo TEST: Q83,820. | Q4,200/Q180 son costos medios simplificados; la cifra mensual usa escala simulada. |
| Recomendación y límites | figures/test_recall_by_mechanism_candidate.png; artefactos/final_evaluation.json | CONSERVAR A; las secuencias contienen señal pero no añadieron valor económico frente a A. | A tuvo recall 53.85% en amount_anomaly y falta validación con datos bancarios reales. |
