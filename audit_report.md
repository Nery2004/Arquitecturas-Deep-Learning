# Auditoría final del Proyecto 1

Fecha de cierre técnico: 2026-09-04. La auditoría verifica artefactos existentes; no reentrena ni modifica decisiones experimentales.

## Datos y protocolo temporal — 15 pts

Evidencias: Dataset Version 1 con SHA-256 verificable; split global por timestamp; `TRAIN 2025-01-01 04:00:48–2025-05-07 06:01:10`, `VALIDATION 2025-05-07 06:10:15–2025-06-03 02:43:40`, `TEST 2025-06-03 02:55:10–2025-06-29 23:59:59`; doce pruebas de preprocessing confirman historia anterior al target, universo común y fit solo con TRAIN.

Riesgos: los datos son sintéticos y la validez depende de los mecanismos definidos por el generador.

Estado: **PASS**.

## Núcleo A y B — 20 pts

Evidencias: checkpoints, metadata, scores TRAIN/VALIDATION/TEST, AP común mediante `average_precision_score`, arquitectura tabular A y GRU B documentadas. A Validation AP 0.910021; B 0.720674.

Riesgos: A y B usan representaciones distintas por diseño; A fue serializado en un entorno distinto del de PyTorch.

Estado: **PASS**, con observación de compatibilidad reproducible.

## Valor del orden — 20 pts

Evidencias: permutación controlada con diez semillas mantuvo eventos, targets y checkpoint. AP cayó de 0.720674 a 0.284325 ± 0.020168. La historia recortada a tres eventos obtuvo 0.710823.

Riesgos: la sensibilidad al orden no implica que B supere a A ni prueba causalidad sobre fraude real.

Estado: **PASS**.

## Apuesta C — 15 pts

Evidencias: hipótesis y criterio previos; C entrenado desde cero contra B; Validation AP 0.815180; ablación con agregados neutralizados 0.429477; costo TEST Q147,780.

Riesgos: C mejoró B, pero no A, y no cumplió el criterio económico frente al mejor baseline.

Estado: **PASS** por ejecución y reporte honesto; la hipótesis tuvo cumplimiento parcial.

## Umbral, costo y recomendación — 15 pts

Evidencias: thresholds elegidos únicamente con VALIDATION; fórmula central `FN × Q4,200 + FP × Q180`; snapshot pre-TEST; evaluación TEST única; candidato A; recomendación CONSERVAR; normalización a 30.44 días.

Riesgos: costos medios simplificados y recall de A de 53.85% para `amount_anomaly`.

Estado: **PASS**.

## Comunicación y reproducibilidad — 15 pts

Evidencias: notebook ejecutado desde kernel limpio, README definitivo, contratos, fingerprints, resultados maestros, matriz de evidencias, tests y dos requirements compatibles con los checkpoints congelados.

Riesgos: falta sustituir el marcador `apellidos` y completar integrantes cuando se conozcan; informe y presentación pertenecen a la etapa siguiente.

Estado: **REVIEW** hasta completar identidad del equipo y entregables posteriores.

## Penalizaciones automáticas

- -20 split aleatorio: **PASS** — no existe split aleatorio; se conservan bloques temporales.
- -15 preprocessing con dataset completo: **PASS** — scalers y vocabularios registran `fit_split=train`.
- -15 accuracy principal: **PASS** — la métrica común es Average Precision; accuracy no dirige decisiones.
- -10 decisiones mirando TEST: **PASS** — snapshot SHA-256 pre-TEST y thresholds de VALIDATION verificables.
- -10 afirmar orden sin permutación: **PASS** — existen diez permutaciones controladas y una prueba de historia corta.

## Inventario lógico

| Responsabilidad | Archivo o ubicación |
|---|---|
| Generación de datos | `src/data_generator.py` |
| Preprocessing causal | `src/preprocessing.py` |
| Entrenamiento A/B/C | `src/train_model_a.py`, `src/train_model_b.py`, `src/train_model_c.py` |
| Falsificaciones | `src/falsification.py` |
| Costos y decisiones | `src/economics.py`, `src/select_thresholds.py` |
| Evaluación final | `src/final_evaluation.py` |
| Modelos y metadata | `artefactos/model_{a,b,c}/` |
| Resultados | `artefactos/final_results.json`, `experiments/` |
| Figuras | `figures/` |
| Notebook | `proyecto1_apellidos.ipynb` |

## Integridad y fingerprints

- Dataset: `1f659a437a417e08b4274da79bfba8853887b2d4888d235c87e4a5d4ce5cf95d`.
- Procesado: `cd496e81ad37906e6da3c4abfdc249da48623a37d807fba1189c4fa198d4b65a`.
- Split: `d938f54ba6258ba09db12412b9bfd7a99d3472fffd68174c53d4a19c2a10f04f`.
- Modelo A: `47a879faa8f4abf4472486584efd6ebf304397dbdb7e87a20ee8ca61f06e16eb`.
- Modelo B: `7162428747420dd68e3e9fc28d800b2d9e43182b40319277bd69ae9f58a3e3b7`.
- Modelo C: `26f0d7e23383aba5fa8c8de3e2cb2609648d4a1f92973a16ea3843d123d12a0e`.
- Configuración congelada antes de TEST: `86c7f7f6bbf9cff302d69fa840c1157d43d4710106023cbd13170b1e8fa21d98`.

## Problemas y acciones

No se encontró una violación que invalide resultados. Se corrigieron textos obsoletos de documentación. La incompatibilidad entre el checkpoint A (NumPy 2.4/sklearn 1.9) y el entorno B/C (NumPy 1.26/sklearn 1.5/PyTorch 2.2) permanece documentada y se resuelve con entornos separados, sin reserializar modelos después de TEST.
