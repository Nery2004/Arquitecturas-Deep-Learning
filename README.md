# Proyecto 1 — Monitoreo Transaccional

## Detectar lo que el orden revela

Este proyecto estudia si leer el orden cronológico de las transacciones mejora la detección de fraude frente a un modelo que usa resúmenes causales del historial. Comparamos un baseline tabular A, una GRU secuencial B y un híbrido C bajo el mismo split temporal, métrica y universo de ejemplos.

**Repositorio del proyecto:** [https://github.com/Nery2004/Arquitecturas-Deep-Learning](https://github.com/Nery2004/Arquitecturas-Deep-Learning)

El resultado fue honesto: B sí usa el orden, pero A obtuvo mejor Average Precision y menor costo. Por eso el candidato final es A y la recomendación es **CONSERVAR** el enfoque agregado.

## Estructura del repositorio

```text
.
├── proyecto1_apellidos.ipynb       # narrativa ejecutada y resultados cargados
├── src/                             # generación, preprocessing, modelos y evaluación
├── data/generated/                  # Dataset Version 1 y configuración
├── data/processed/                  # splits e inputs transformados
├── artefactos/                      # preprocessing, checkpoints y resultados maestros
├── experiments/                     # comparaciones, thresholds y errores
├── figures/                         # figuras reproducibles
├── tests/                           # integridad, modelos y cierre final
├── evidence_matrix.csv
├── audit_report.md
├── requirements.txt
└── requirements-model-a.txt
```

## Datos

Se usó la Ruta A: datos sintéticos propios y reproducibles. Dataset Version 1 contiene 95,767 transacciones de 2,800 tarjetas entre 2025-01-01 y 2025-06-29, con 1.6446% de fraude y semilla 42. Incluye `testing_cashout`, `channel_takeover`, `amount_anomaly` y hard negatives como viajes, compras grandes, shopping sprees y microcompras legítimas.

| Split | Inicio | Fin | Ejemplos | Fraudes |
|---|---|---|---:|---:|
| TRAIN | 2025-01-01 04:00:48 | 2025-05-07 06:01:10 | 64,236 | 1,110 |
| VALIDATION | 2025-05-07 06:10:15 | 2025-06-03 02:43:40 | 14,365 | 228 |
| TEST | 2025-06-03 02:55:10 | 2025-06-29 23:59:59 | 14,366 | 222 |

No hubo shuffle. Cada historia contiene solo eventos anteriores al target. Scalers, imputación y vocabularios se ajustaron exclusivamente con TRAIN.

## Modelos

| Modelo | Información | Arquitectura |
|---|---|---|
| A | Operación actual + 23 agregados causales | HistGradientBoosting |
| B | Operación actual + hasta 12 eventos ordenados | GRU unidireccional |
| C | Operación actual + secuencia + agregados | GRU híbrida de tres ramas |

La métrica principal común es Average Precision mediante `sklearn.metrics.average_precision_score`. Accuracy no se usa para seleccionar modelos.

## Resultados

| Modelo | Validation AP | Test AP | Threshold | Precision | Recall | F1 | Costo TEST |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 0.910021 | 0.919108 | 0.055489 | 0.5024 | 0.9505 | 0.6573 | Q83,820 |
| B | 0.720674 | 0.715009 | 0.825399 | 0.3023 | 0.8784 | 0.4498 | Q194,400 |
| C | 0.815180 | 0.824840 | 0.668239 | 0.3778 | 0.9054 | 0.5332 | Q147,780 |

Cada threshold minimizó `FN × Q4,200 + FP × Q180` exclusivamente en VALIDATION. La configuración se congeló antes de abrir TEST. El equivalente mensual de A es Q94,927.01 usando 30.44 días, dentro de la escala simulada; no se extrapola a 1.4 millones de tarjetas.

## ¿El orden aporta?

B original obtuvo AP 0.720674 en VALIDATION. Al permutar los mismos eventos dentro de cada historia, la media cayó a 0.284325 con desviación 0.020168: caída absoluta 0.436349. Esto demuestra que B utiliza información relacionada con el orden.

Al conservar solo los tres eventos más recientes, AP fue 0.710823. El orden importa para B, pero el contexto entre los eventos cuarto y duodécimo aportó poco y B no superó a A.

## Apuesta C

La hipótesis previa fue que combinar la secuencia de B con los agregados de A mejoraría la detección. C alcanzó 0.815180 AP frente a 0.720674 de B; al neutralizar los agregados cayó a 0.429477. La parte predictiva se cumplió, pero C quedó debajo de A y costó Q63,960 más en TEST. El veredicto completo es **cumplimiento parcial, sin ventaja operativa final**.

## Decisión final

- Candidato: **Modelo A**.
- Artefacto: `artefactos/model_a/model_a.joblib`.
- Threshold: `0.05548898134596948`.
- Recomendación: **CONSERVAR**.
- Limitación principal observada: recall de 53.85% para `amount_anomaly` en TEST.

## Reproducción

### Entorno principal

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Si el Python local no incluye `pip`, puede usarse `uv pip install --python .venv/bin/python -r requirements.txt`.

### Reconstrucción experimental completa

Estos comandos regeneran artefactos y deben utilizarse solo en una copia de trabajo, porque el experimento entregado está cerrado:

```bash
.venv/bin/python -m src.data_generator
.venv/bin/python -m src.preprocessing
python3 -m src.train_model_a
.venv/bin/python -m src.train_model_b
.venv/bin/python -m src.falsification
.venv/bin/python -m src.train_model_c
.venv/bin/python -m src.select_thresholds
.venv/bin/python -m src.final_evaluation
```

Modelo A fue congelado con NumPy 2.4.4 y scikit-learn 1.9.0; su entorno compatible se especifica en `requirements-model-a.txt`. B/C usan el entorno principal con PyTorch 2.2.2. Esta separación evita reserializar checkpoints después de TEST.

### Reproducir resultados sin volver a entrenar

```bash
python3 -m json.tool artefactos/final_results.json
python3 -m json.tool artefactos/final_candidate.json
.venv/bin/python -m pytest -q --ignore=tests/test_model_a.py
python3 -m pytest -q tests/test_model_a.py
.venv/bin/jupyter nbconvert --to notebook --execute --inplace proyecto1_apellidos.ipynb
.venv/bin/python informe/build_report.py
```

El notebook carga JSON/CSV/PNG ya congelados; no entrena ni vuelve a evaluar TEST.

## Semilla y fingerprints

- Seed: `42`.
- Dataset: `1f659a437a417e08b4274da79bfba8853887b2d4888d235c87e4a5d4ce5cf95d`.
- Procesado: `cd496e81ad37906e6da3c4abfdc249da48623a37d807fba1189c4fa198d4b65a`.
- Split: `d938f54ba6258ba09db12412b9bfd7a99d3472fffd68174c53d4a19c2a10f04f`.
- A: `47a879faa8f4abf4472486584efd6ebf304397dbdb7e87a20ee8ca61f06e16eb`.
- B: `7162428747420dd68e3e9fc28d800b2d9e43182b40319277bd69ae9f58a3e3b7`.
- C: `26f0d7e23383aba5fa8c8de3e2cb2609648d4a1f92973a16ea3843d123d12a0e`.

## Tres decisiones técnicas importantes

### 1. GRU

- Decisión: usar una GRU unidireccional para B.
- Alternativas: RNN simple, LSTM y Transformer.
- Por qué: equilibrio entre memoria, parámetros y facilidad de explicación para secuencias cortas.
- Evidencia: AP 0.720674; al permutar el orden cayó a 0.284325.
- Veredicto: la GRU usa el orden, pero no supera al baseline agregado.

### 2. Split temporal

- Decisión: TRAIN/VALIDATION/TEST global por timestamp.
- Alternativa: split aleatorio.
- Por qué: simula predecir el futuro y evita mezclar objetivos posteriores con anteriores.
- Evidencia: fronteras estrictas verificadas y doce controles de preprocessing.
- Veredicto: decisión conservada; no se encontró leakage temporal.

### 3. Modelo C híbrido

- Decisión: fusionar GRU, operación actual y agregados.
- Alternativas: solo secuencia o ensemble tardío.
- Por qué: orden y resúmenes históricos podían ser complementarios.
- Evidencia: C mejoró B en 0.094507 AP; la ablación confirmó uso de agregados, pero C costó más que A.
- Veredicto: hipótesis predictiva parcialmente respaldada, sin justificación económica para sustituir A.

## Candidato al Proyecto Final

- Modelo conservado: A.
- Ubicación: `artefactos/model_a/model_a.joblib`.
- Usuario del score: motor de autorización y equipo de riesgos.
- Decisión: bloquear cuando `risk_score >= 0.05548898134596948`; antes de producción se requiere definir revisión humana y excepciones.
- Entrada: operación actual más historia causal de la misma tarjeta, transformada a current features y 23 agregados.
- Salida: `risk_score`, `threshold` y `decision` (`allow` o `block`).
- Contrato: `artefactos/input_output_contract.json`.
- Límites y riesgos: datos sintéticos, falsos bloqueos, fraude no detectado, drift, costos simplificados y bajo recall de `amount_anomaly`.
- Datos faltantes: transacciones reales autorizadas, costos operativos validados, latencia, capacidad de revisión, gobernanza y seguimiento de drift.

## Limitaciones

- Los datos y mecanismos de fraude fueron diseñados; no representan toda la conducta bancaria real.
- La estimación económica corresponde a la escala simulada y usa costos promedio simplificados.
- MAX_HISTORY=12 limita lo que B/C pueden observar; la prueba corta no estudia horizontes mayores.
- Existen 209 falsos positivos y 11 falsos negativos de A en TEST.
- El comportamiento puede cambiar con el tiempo; se necesita monitoreo de drift.
- Antes de bloquear transacciones reales se requiere validación externa, revisión de impacto y datos bancarios autorizados.

## Uso de inteligencia artificial

Se utilizó IA como apoyo para estructurar el trabajo, revisar código, discutir alternativas, depurar y mejorar la documentación. Los integrantes verificaron el código, los artefactos, las métricas, las decisiones y la interpretación. La responsabilidad académica y la defensa de los resultados corresponden al equipo.

La auditoría completa está en `audit_report.md` y las seis evidencias principales en `evidence_matrix.csv`.
