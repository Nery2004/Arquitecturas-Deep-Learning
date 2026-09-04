# Proyecto 1 — Monitoreo transaccional

Proyecto del curso **Deep Learning 2026**. El objetivo es medir si el orden cronológico de las transacciones aporta información para detectar fraude que no esté ya contenida en variables agregadas, y estimar cuánto vale esa información en términos económicos.

Este repositorio contiene las **etapas 0 a 3**: diseño experimental y Dataset Version 1. No se han entrenado modelos, creado splits, elegido umbrales ni calculado métricas predictivas.

## Ruta de datos

Usamos la **Ruta A: datos sintéticos con generador propio**. El generador reproducible contiene varios perfiles legítimos, casos legítimos difíciles y tres mecanismos de fraude. Uno depende principalmente del orden. Esta elección permite controlar qué señal existe y comprobar más adelante si un modelo aprovecha la secuencia, sin presentar datos sintéticos como si fueran comportamiento bancario real.

## Datos sintéticos

Dataset Version 1 contiene **95,767 transacciones de 2,800 tarjetas**, desde `2025-01-01T00:00:00` hasta `2025-06-29T23:59:59`. Se generó con semilla 42 y una prevalencia de fraude de **1.6446 %**. Las transacciones crudas están en `data/generated/transactions.csv`; la configuración exacta en `data/generated/generator_config.json`; y el resumen, validaciones y hash en `data/generated/dataset_metadata.json`.

El formato es CSV porque el entorno actual no cuenta con pandas ni un motor Parquet. Esto evita instalar dependencias solo para serializar y hace que el fingerprint corresponda a los bytes exactos que consumirán las etapas posteriores.

Las columnas son `transaction_id`, `card_id`, `timestamp`, `amount`, `merchant_category`, `channel`, `distance_from_home_km`, `is_international`, `is_fraud`, `fraud_type`, `fraud_stage`, `customer_profile` y `hard_negative_type`. Los IDs, etiquetas, etapas, perfil y marca de hard negative están declarados centralmente como metadata y **no son features**. Las variables temporales derivadas se calcularán después de forma causal.

Los perfiles son regular, online, alto gasto, variable y viajero. Sus preferencias latentes modifican monto típico, dispersión, frecuencia, horario, categorías, canales y distancia. Ninguna de esas preferencias internas se entrega directamente como predictor.

Los mecanismos generados son:

- `testing_cashout`: de dos a cinco pruebas pequeñas seguidas por un cashout plausible.
- `channel_takeover`: transición breve hacia canales poco habituales para la tarjeta, con contexto geográfico e internacional variable.
- `amount_anomaly`: monto claramente desviado del comportamiento individual, sin exigir un orden particular.

Los hard negatives incluyen viajes legítimos, compras grandes válidas, shopping sprees y microcompras consecutivas. Una compra legítima grande durante un viaje puede parecer una toma de cuenta; en sentido contrario, un fraude adaptado al comportamiento normal puede no dejar señal suficiente.

Fingerprint SHA-256 del Dataset Version 1:

`1f659a437a417e08b4274da79bfba8853887b2d4888d235c87e4a5d4ce5cf95d`

## Dataset congelado

Esta versión se congeló antes del entrenamiento. Los mecanismos y la configuración principal no se modificarán después para favorecer al Modelo A, B o C. Un cambio solo será válido ante un bug real y dará lugar a una versión nueva con su motivo y fingerprint documentados.

El generador creó una única línea temporal completa sin conocer fronteras. Los splits descritos a continuación se definieron después y no modifican Dataset Version 1.

## Partición temporal

La pertenencia la determina el timestamp de la transacción objetivo. No se hizo shuffle ni estratificación:

| Split | Inicio | Fin | Targets | Fraudes | Tasa |
|---|---|---|---:|---:|---:|
| TRAIN | 2025-01-01 04:00:48 | 2025-05-07 06:01:10 | 64,236 | 1,110 | 1.7280% |
| VALIDATION | 2025-05-07 06:10:15 | 2025-06-03 02:43:40 | 14,365 | 228 | 1.5872% |
| TEST | 2025-06-03 02:55:10 | 2025-06-29 23:59:59 | 14,366 | 222 | 1.5453% |

Las fronteras conservan completo cada timestamp y cumplen `max(TRAIN) < min(VALIDATION) < max(VALIDATION) < min(TEST)`. Si una tarjeta tiene operaciones con el mismo timestamp, `transaction_id` actúa como desempate determinista. Un objetivo de VALIDATION o TEST puede usar operaciones anteriores de splits previos como historia observable; sus etiquetas nunca entran como features.

El split quedó congelado en `data/processed/split_config.json` y todos los modelos deberán cargarlo mediante la misma función.

## Preprocessing

Cada ejemplo es una transacción objetivo con al menos una operación previa. Se excluye únicamente la primera transacción de cada tarjeta: 2,800 filas. El universo común contiene 92,967 ejemplos en `data/processed/example_index.csv`.

La operación actual se representa por monto, distancia, internacionalidad, hora y día cíclicos, tiempo desde la operación anterior, comercio y canal. El Modelo A añadirá 23 agregados causales: estadísticas históricas personales, razones de monto y distancia, frecuencias y flags de canal/comercio inusual, conteos de 1h/6h/24h/7d y resúmenes de las últimas 24 horas. No recibe posiciones anteriores ni una secuencia disfrazada.

Para B y C, `history_sequence` contiene hasta las últimas 12 operaciones **anteriores**, en orden ascendente. Cada evento tiene ocho numéricas (`amount`, distancia, internacionalidad, hora/día cíclicos y tiempo desde el evento anterior) y dos categóricas (comercio y canal). Se usa left padding: `PAD=0`, `UNK=1`, categorías reales desde 2, junto con máscara y longitud válida. La operación actual permanece separada. Cambiar posteriormente 12 por 3 no requiere reconstruir la definición del target.

Los tres escaladores aplican imputación por mediana y estandarización; todos se ajustaron exclusivamente con TRAIN. El delta desconocido del primer evento histórico y resúmenes 24h sin eventos se imputan con medianas TRAIN. Conteos, diversidad y flags tienen valores conceptuales de cero. Los vocabularios también se aprendieron solo de TRAIN.

## Controles contra fuga de información

- Fingerprint del CSV verificado antes de procesar.
- Split estricto por timestamp del target, con fronteras congeladas.
- Historia limitada a la misma tarjeta y anterior al target; `transaction_id` resuelve empates.
- Agregados recalculados sobre historia anterior en muestras de control.
- `is_fraud`, `fraud_type`, `fraud_stage`, `hard_negative_type`, perfil e IDs excluidos de inputs.
- Encoders, imputadores y escaladores ajustados solo con TRAIN.
- A, B y C comparten `example_index` y `y`.
- Secuencia ascendente, left padding y máscara comprobados automáticamente.
- TEST fue transformado con parámetros congelados, pero no se usó para decisiones.

## Datos procesados

- `data/processed/example_index.csv`: fuente de verdad de targets, splits y metadata de análisis.
- `data/processed/aggregate_features_raw.csv`: agregados causales sin escalar para auditoría.
- `data/processed/model_inputs_{train,validation,test}.npz`: entradas transformadas comunes.
- `data/processed/split_config.json`: fronteras y fingerprint fuente.
- `data/processed/processed_metadata.json`: definición, EDA, validaciones y fingerprint procesado.
- `artefactos/preprocessing/*.json`: scalers y vocabularios TRAIN-only.

Para reproducir: `python3 -m src.preprocessing`. El fingerprint procesado es `cd496e81ad37906e6da3c4abfdc249da48623a37d807fba1189c4fa198d4b65a` y coincidió en dos ejecuciones consecutivas.

## Modelo A — Baseline sin orden

Antes de probar una red recurrente medimos qué puede lograrse sin leer el orden completo. Modelo A recibe las 8 variables numéricas y 2 categóricas de la operación actual, más los 23 agregados históricos ya congelados. No recibe secuencias, posiciones, IDs ni metadata del generador.

Se comparó una regresión logística de sanity check con cinco configuraciones moderadas de `HistGradientBoostingClassifier`. El desbalance se trató mediante pesos calculados solo desde TRAIN: 0.50879 para legítimas y 28.93514 para fraude. No hubo resampling. La métrica común quedó fijada como **AUC-PR / Average Precision (AP)** mediante `average_precision_score`.

El candidato congelado es `hgb_02`: learning rate 0.08, 180 iteraciones, 31 hojas máximas, mínimo 40 observaciones por hoja y regularización L2 de 1.0. Obtuvo AP 0.999970 en TRAIN y 0.910021 en VALIDATION. El gap de 0.089948 muestra sobreajuste y debe considerarse al comparar con B. `hgb_05` obtuvo 0.910294 en VALIDATION, solo 0.000273 más; se prefirió `hgb_02` por menor complejidad conforme a la regla previa de tratar diferencias menores a 0.001 como empate práctico.

El modelo está en `artefactos/model_a/model_a.joblib`, su metadata en `artefactos/model_a/model_a_metadata.json` y la matriz de experimentos en `experiments/model_a_results.csv`. La curva Precision–Recall usa solamente VALIDATION. No existen scores ni métricas de TEST.

## Modelo B — GRU secuencial

Modelo B recibe hasta 12 operaciones anteriores, ordenadas de la más antigua a la más reciente, y las características observables de la operación actual. No carga los 23 agregados de A. Cada evento combina ocho variables numéricas con embeddings de comercio (6 dimensiones) y canal (3 dimensiones). El left padding se compacta antes de `pack_padded_sequence`, por lo que la GRU ignora PAD y el hidden state corresponde al último evento válido.

La arquitectura congelada (`b3`) usa una GRU unidireccional de una capa y 64 unidades. La rama actual proyecta sus variables a 64 unidades; ambas representaciones se concatenan y pasan por Dense/ReLU, dropout 0.4 y una salida lineal. Durante fit se usan logits con `BCEWithLogitsLoss`, `pos_weight=56.87027` calculado solo con TRAIN, AdamW, learning rate 0.0008 y batch 256. Tiene 25,499 parámetros entrenables.

Early stopping observó exclusivamente AP de VALIDATION, con paciencia 5. El mejor checkpoint fue epoch 9; se ejecutaron 14 epochs. El resultado congelado es AP 0.836821 en TRAIN y 0.720674 en VALIDATION, con gap 0.116147. Se ejecutó en CPU para priorizar determinismo y funciona sin GPU.

Modelo A obtuvo 0.910021 y B 0.720674 en la misma VALIDATION. Esta diferencia por sí sola **no demuestra** que el orden aporte o no aporte información: todavía debemos destruir el orden manteniendo los eventos y ejecutar las falsificaciones predefinidas. No se generaron predicciones de TEST.

El checkpoint está en `artefactos/model_b/model_b.pt`, la metadata en `artefactos/model_b/model_b_metadata.json`, el historial en `artefactos/model_b/training_history.csv` y los candidatos en `experiments/model_b_results.csv`. Se reproduce con `.venv/bin/python -m src.train_model_b`.

## Evidencia del valor del orden

Para comprobar que B no dependía solamente de los eventos vistos como un conjunto, barajamos los eventos válidos dentro de cada historia de VALIDATION. Montos, canales, categorías, distancias, variables temporales, padding, longitudes, operación actual, target y checkpoint permanecieron iguales. `time_since_previous` viajó pegado a su evento; no se recalculó. Se evaluaron seeds 100–109 sin reentrenar.

El AP original fue 0.720674. Las permutaciones obtuvieron una media de 0.284325, desviación 0.020168 y rango 0.255316–0.326197. La caída absoluta fue 0.436349 y la relativa 60.55%. Esta caída grande y consistente es evidencia fuerte de que B utiliza información relacionada con el orden. No demuestra causalidad ni hace que B supere a A.

En evaluación one-vs-legitimate, `testing_cashout` cayó de AP 0.802213 a aproximadamente 0.200 tras permutar; `channel_takeover`, de 0.525874 a aproximadamente 0.199. `amount_anomaly` tenía AP 0.006153 y permaneció muy bajo, coherente con que B no lo representa bien. Los demás fraudes se excluyeron al calcular cada AP específico.

## Prueba de historia recortada

Usamos el mismo checkpoint y mantuvimos la shape 12, pero dejamos únicamente los tres eventos reales más recientes. AP pasó de 0.720674 a 0.710823: caída absoluta 0.009850 y relativa 1.37%. Los 14,365 ejemplos de VALIDATION tenían más de tres eventos disponibles, de modo que todos fueron intervenidos.

La evidencia del valor de contexto más allá de tres operaciones es limitada para este modelo. `testing_cashout` bajó de AP 0.802213 a 0.781959; `channel_takeover`, de 0.525874 a 0.521238; y `amount_anomaly`, de 0.006153 a 0.005263. Permutar pregunta si importa el orden de los mismos eventos; recortar pregunta si importa disponer de una historia más larga. Son pruebas distintas.

Las evidencias completas están en `artefactos/model_b/falsification_metadata.json` y `experiments/falsification_results.csv`. TEST no se utilizó.

## Apuesta C — Modelo híbrido

La hipótesis escrita antes de entrenar fue que combinar secuencia y agregados mejoraría la detección porque cada representación resume un aspecto distinto. El control es B. El criterio previo exige que C supere el AP de B en VALIDATION y, posteriormente, que la mejora no aumente el costo económico.

C se entrenó desde cero. Conserva la GRU de B —64 unidades, una capa, embeddings 6/3 y rama current de 64— y añade una rama `Linear(23, 32) + ReLU + Dropout(0.2)` para los agregados históricos de A. Las tres ramas se concatenan y pasan por una fusión de 64 unidades, ReLU, dropout 0.4 y una salida lineal. No se duplicaron las current features dentro de la rama agregada.

El candidato C2 alcanzó AP 0.881519 en TRAIN y 0.815180 en VALIDATION, con gap 0.066339. Superó a B por 0.094507 AP (+13.11%), aunque quedó debajo de A (0.910021). Por tanto, C cumple la parte predictiva del criterio respecto al control B; el veredicto completo es parcial porque la condición económica todavía no se ha evaluado.

Al neutralizar los agregados estandarizados con cero usando el mismo checkpoint, AP cayó de 0.815180 a 0.429477. Esto indica que C depende de la rama agregada. Cero representa la media de TRAIN después del escalado, no valores aleatorios. El checkpoint quedó congelado en `artefactos/model_c/model_c.pt`; TEST no se evaluó.

## Pregunta de investigación

¿El orden cronológico de las transacciones aporta información adicional para detectar fraude que no pueda capturarse únicamente mediante variables agregadas?

La comparación busca aislar el valor del orden. No basta con que una GRU obtenga una métrica alta: debe superar una referencia sin secuencia bajo condiciones comparables y reaccionar a pruebas que destruyen o reducen el contexto temporal.

## Unidad y momento de predicción

Cada ejemplo es una **transacción objetivo de una tarjeta**. Cuando la operación se intenta realizar, el sistema estima su riesgo usando solamente:

- las características conocidas de la operación actual (por ejemplo, monto, canal, comercio, hora y ubicación o distancia disponible en la solicitud); y
- el historial de esa tarjeta estrictamente anterior a la operación.

La salida de todos los modelos será `risk_score ∈ [0, 1]`. La etiqueta `is_fraud`, el campo auxiliar `fraud_type`, cualquier dato creado después de la decisión y el resultado de transacciones futuras quedan fuera de las entradas. La decisión operativa se toma antes de conocer la etiqueta.

Para mantener una comparación justa, el Modelo A puede ver la operación actual y resúmenes causales del historial; el Modelo B puede ver la misma operación actual y una ventana ordenada del historial. Esta convención evita darle a B información actual que A no recibe. Inicialmente probaremos **12 eventos anteriores más la transacción objetivo como paso actual**. Doce es un punto de partida manejable que puede cubrir ráfagas cortas sin imponer secuencias costosas. La longitud podrá cambiar únicamente con TRAIN y VALIDATION; nunca con TEST.

## Partición temporal y protección de TEST

El punto de partida es una división global por tiempo, sin shuffle:

- TRAIN: primer 70 % de las transacciones objetivo.
- VALIDATION: siguiente 15 %.
- TEST: último 15 %.

Se exigirán las condiciones `max(timestamp_train) < min(timestamp_validation)` y `max(timestamp_validation) < min(timestamp_test)`. Los empates de timestamp en una frontera se resolverán moviendo el bloque completo a un solo split, aunque los porcentajes cambien ligeramente. Así, una misma marca temporal no aparece a ambos lados de una frontera.

El historial que acompaña a un ejemplo puede contener eventos anteriores a la frontera —eso sería información disponible en producción—, pero la transacción objetivo pertenece a un único split. Ninguna etiqueta futura puede entrar en las características. También se revisará que una tarjeta nueva o un historial corto se representen de manera explícita, no descartando casos de forma distinta entre modelos.

**TEST permanecerá cerrado hasta el final.** No se usará para seleccionar variables, ajustar normalizadores o codificadores, elegir longitud de secuencia, arquitectura, hiperparámetros, Modelo C, threshold ni para decidir si una hipótesis funciona. Todas esas decisiones se tomarán con TRAIN y VALIDATION. Después se congelarán el pipeline, el modelo y el threshold; TEST se evaluará una sola vez.

## Política contra data leakage

Cualquier transformación que aprenda parámetros se ajustará exclusivamente con TRAIN: `Scaler.fit(TRAIN)`, `Encoder.fit(TRAIN)` y cálculos estadísticos equivalentes. Luego se aplicará `transform` por separado a TRAIN, VALIDATION y TEST. No se hará `fit(DATASET_COMPLETO)`.

Los agregados de cada ejemplo usarán ventanas causales: incluirán solo eventos anteriores a la transacción objetivo. Las variables de la operación actual se añadirán aparte. No se usarán estadísticas futuras, totales calculados al final de la vida de la tarjeta, etiquetas históricas que no estuvieran confirmadas en ese momento, `is_fraud` ni `fraud_type`. Las categorías desconocidas deberán manejarse sin volver a ajustar el encoder fuera de TRAIN.

## Diseño de los modelos

### Modelo A — baseline sin orden

Será un modelo de boosting basado en árboles disponible en scikit-learn, inicialmente `HistGradientBoostingClassifier` sobre una representación numérica correctamente preparada. Recibirá características de la operación actual y agregados causales como promedio, desviación y máximo reciente del monto; conteos y frecuencia reciente; diversidad de comercios y canales; tiempo desde la transacción anterior; y razón entre el monto actual y el promedio histórico.

No recibirá una lista ordenada de eventos, posiciones, transiciones completas ni una concatenación que permita reconstruir la secuencia. Puede recibir resúmenes temporales legítimos —por ejemplo, frecuencia reciente— porque el objetivo es compararlo con un sistema antifraude competitivo, no con un baseline debilitado.

### Modelo B — GRU secuencial

Flujo conceptual: eventos ordenados → representación numérica → GRU → capa densa → sigmoid → `risk_score`.

Comenzamos con una GRU porque está diseñada para secuencias, puede conservar contexto, tiene menos parámetros que alternativas más pesadas y resulta clara de explicar. Consideramos RNN simple, LSTM, CNN temporal y Transformer. No asumimos que la GRU sea universalmente mejor: es un equilibrio inicial entre capacidad, costo y claridad experimental.

### Modelo C — híbrido (apuesta del equipo)

Una GRU producirá la representación secuencial y una red densa pequeña procesará los agregados. Ambas representaciones se concatenarán antes de las capas de salida.

**Hipótesis previa:** creemos que C puede mejorar porque la secuencia representa transiciones y patrones de orden, mientras los agregados resumen el comportamiento reciente. Son vistas distintas y potencialmente complementarias.

**Criterio de éxito previo a TEST:** en VALIDATION, con el mismo split y protocolo, C será candidato útil solo si su AUC-PR es mayor que la de B y, al seleccionar para cada modelo su threshold con la misma regla de costo en VALIDATION, su costo económico total no es mayor que el de B. No fijamos una mejora porcentual sin evidencia. Si la diferencia es muy pequeña, se estimará su incertidumbre con bootstrap por tarjeta en VALIDATION; el intervalo de la diferencia deberá respaldar una mejora y no solo ruido. Esta regla podrá detallarse antes de ejecutar comparaciones, pero no después de mirar TEST.

## Métricas y costo económico

La métrica principal independiente del threshold será **AUC-PR**, adecuada para clases desbalanceadas. En el threshold elegido únicamente con VALIDATION se reportarán precision, recall y F1. Accuracy, si se muestra, será una referencia secundaria y nunca el criterio principal.

Para un threshold dado:

`economic_cost = false_negatives × Q4,200 + false_positives × Q180`

El threshold se elegirá en VALIDATION con una regla reproducible de minimización de este costo, se congelará y después se aplicará una sola vez a TEST. Aún no existe ni se ha escogido ningún threshold. Además del costo total se reportará una versión normalizada por número de transacciones, porque el total depende del tamaño del split.

## Pruebas de falsificación declaradas antes del entrenamiento

1. **Permutación controlada.** Se barajará el orden dentro de cada secuencia de evaluación manteniendo eventos, valores, longitud, agregados y etiqueta. Se comparará B sobre secuencias originales y permutadas. La permutación será reproducible y se repetirá con varias semillas definidas antes de abrir TEST. Una caída respaldaría que B usa el orden; si no cae, no afirmaremos que lo usa.
2. **Recorte de historia.** Se comparará el contexto normal de 12 eventos anteriores con uno corto de 3, sin cambiar el universo de ejemplos. La predicción previa es que los fraudes que necesitan una cadena de acciones perderán detectabilidad al reducir el contexto. La comparación principal y los ajustes se harán en VALIDATION antes del uso final de TEST.

## Diseño preliminar de datos sintéticos

Variables propuestas del evento: `timestamp`, `card_id`, `amount`, `merchant_category`, `channel`, `hour`, `day_of_week`, `time_since_previous` y `distance_from_previous`. `is_fraud` será la etiqueta y `fraud_type` servirá solo para auditoría y análisis estratificado; ninguno entrará como predictor. `hour` y `day_of_week` deberán derivarse causalmente del timestamp, y la distancia no podrá calcularse con una ubicación posterior.

Mecanismos previstos:

- **Card testing → cashout:** varias operaciones pequeñas cercanas en el tiempo seguidas por una operación grande. El orden será parte esencial de la señal; los mismos eventos reordenados no deben equivaler al patrón.
- **Cambio anormal de canal o comportamiento:** transición poco habitual respecto al perfil, por ejemplo ONLINE desconocido → ATM en poco tiempo. Combina canal, transición, tiempo e historial.
- **Anomalía de monto:** una compra extraordinariamente grande respecto a un rango estable. Debe ser razonablemente detectable mediante agregados para no favorecer artificialmente al modelo secuencial.

Habrá casos legítimos difíciles: boletos de avión, gastos extraordinarios, vacaciones, cambios válidos de canal, compras seguidas y tarjetas naturalmente irregulares. Esperamos, por ejemplo, que un viaje legítimo con canal nuevo, distancia alta y gasto grande pueda confundirse con fraude. También evitaremos que una sola categoría, canal, rango de monto o periodo temporal revele la etiqueta por sí solo.

La prevalencia y la mezcla de mecanismos se definirán en la etapa 2. Tendrán que producir suficiente fraude para evaluar sin convertir el problema en uno artificialmente balanceado. Se auditarán tasas por tiempo, tarjeta y tipo para detectar atajos involuntarios.

## Justicia experimental

A, B y C usarán el mismo universo de transacciones objetivo, etiqueta, horizonte, fronteras temporales y TEST. Los ejemplos con poco historial no se eliminarán selectivamente: usarán padding/máscara en secuencias y agregados definidos de forma coherente. Se compararán con el mismo protocolo y ninguna arquitectura tendrá acceso a información futura o campos auxiliares.

## Reproducibilidad

La semilla central es `RANDOM_SEED = 42`, definida en `src/config.py`. En etapas posteriores se aplicará a `random`, NumPy, el framework de Deep Learning, el generador y los cargadores de datos. Elegimos PyTorch como candidato inicial por su control explícito del entrenamiento y disponibilidad habitual en Colab; la decisión puede revisarse antes de implementar los modelos.

El generador deberá devolver exactamente los mismos datos para la misma semilla, versión del código y parámetros. También se registrarán versiones, configuración, fronteras temporales y semillas de cada ejecución. La reproducibilidad exacta entre hardware puede requerir algoritmos deterministas y documentar operaciones no deterministas.

En el entorno inspeccionado durante las etapas 0 y 1 estaban disponibles NumPy 2.4.4 y matplotlib 3.10.9; pandas, scikit-learn, PyTorch y Jupyter no estaban instalados. `requirements.txt` declara el conjunto mínimo previsto, pero no se instaló nada en esta etapa ni se fijaron versiones antes de comprobar compatibilidad en la etapa de implementación.

## Decisiones técnicas importantes

### 1. GRU como modelo secuencial principal

- **Decisión:** comenzar con una GRU.
- **Alternativas consideradas:** RNN simple, LSTM, CNN temporal y Transformer.
- **Razón inicial:** buen equilibrio entre memoria secuencial, número de parámetros y facilidad de explicación.
- **Evidencia que deberá revisarse posteriormente:** B alcanzó AP 0.720674 en VALIDATION y fue reproducible; quedó debajo de A. Faltan costo y pruebas de falsificación.
- **Veredicto final:** GRU congelada como candidato B; pendiente el veredicto sobre el valor del orden.

### 2. Partición estrictamente temporal

- **Decisión:** 70/15/15 por tiempo, sin shuffle, preservando bloques con el mismo timestamp.
- **Alternativas consideradas:** split aleatorio y partición por tarjeta.
- **Razón inicial:** reproduce una evaluación hacia el futuro y reduce leakage temporal.
- **Evidencia que deberá revisarse posteriormente:** tamaños, prevalencia, cobertura de tarjetas y mecanismos por split.
- **Veredicto final:** pendiente.

### 3. Modelo híbrido como apuesta C

- **Decisión:** combinar representación GRU y agregados.
- **Alternativas consideradas:** GRU sola, ensemble tardío y red densa solo con agregados.
- **Razón inicial:** las dos representaciones pueden capturar señales complementarias.
- **Evidencia que deberá revisarse posteriormente:** C obtuvo AP 0.815180 frente a 0.720674 de B; la ablación sin agregados cayó a 0.429477. Falta el criterio económico.
- **Veredicto final:** cumple la parte predictiva de la apuesta; veredicto completo pendiente de evaluación económica.

### 4. HistGradientBoosting como Modelo A

- **Decisión:** congelar `hgb_02` como baseline sin orden.
- **Alternativas consideradas:** regresión logística y cuatro configuraciones adicionales de HistGradientBoosting.
- **Razón inicial:** relaciones no lineales, interacciones tabulares y soporte nativo de categóricas sin añadir dependencias externas.
- **Evidencia que deberá revisarse posteriormente:** AP 0.910021 en VALIDATION; gap TRAIN–VALIDATION de 0.089948. TEST permanece sellado.
- **Veredicto final:** candidato de Modelo A congelado para compararlo después con B y C.

## Candidato al Proyecto Final

- **Modelo que se conservaría:** pendiente después de la evaluación final.
- **Ruta del artefacto:** pendiente; se ubicará bajo `artefactos/`.
- **Quién utilizaría el `risk_score`:** pendiente de definir con el caso operativo.
- **Qué decisión tomaría:** pendiente; no se asume todavía bloqueo automático.
- **Contrato preliminar de entrada:** pendiente de concretar; deberá incluir características de la operación actual e historial causal según el modelo elegido.
- **Contrato preliminar de salida:** objeto con `risk_score` en `[0, 1]`, versión del modelo y timestamp de inferencia; esquema definitivo pendiente.
- **Principales límites:** datos sintéticos, cambio de distribución, tarjetas con poco historial y diferencias respecto a fraude real.
- **Riesgos:** falsos bloqueos, fraude no detectado, sesgos del generador, leakage y uso fuera del contexto evaluado.
- **Información adicional necesaria:** validación con datos reales autorizados, restricciones operativas, latencia, gobernanza y proceso de revisión humana.

## Estructura

```text
.
├── proyecto1_apellidos.ipynb
├── README.md
├── requirements.txt
├── src/
├── data/generated/
├── artefactos/
├── figures/
├── informe/
└── presentacion/
```

El nombre del notebook conserva `apellidos` como marcador visible hasta conocer los integrantes. Los módulos de `src/` son contratos preliminares, no implementaciones.

## Estado y siguientes etapas

Etapas 0 a 5 completadas. Los datos están preparados para iniciar el Modelo A. Queda pendiente el entrenamiento y toda evaluación predictiva; no se aplicó balanceo ni se entrenó ningún modelo.

Lista de control actual:

- [x] No hay split aleatorio en datos temporales.
- [x] No se ajustó preprocessing sobre el dataset completo.
- [x] No se utilizó TEST.
- [x] No se eligió arquitectura mirando TEST.
- [x] No se escogió threshold.
- [x] Accuracy no es la métrica principal.
- [x] No se afirmó que el orden aporta.
- [x] No se entrenó ningún modelo.
- [x] No se inventaron resultados.
- [x] La hipótesis de C quedó definida antes del entrenamiento.
- [x] Las pruebas de falsificación quedaron declaradas antes de ejecutarse.
- [x] A, B y C tendrán una comparación justa.

## Uso de inteligencia artificial

Se utilizó inteligencia artificial como apoyo para estructurar el proyecto, discutir alternativas, apoyar la implementación, revisar código y mejorar la documentación. Los integrantes son responsables de revisar y poder explicar las decisiones experimentales, comprobar el funcionamiento del código y validar los resultados y su interpretación. La IA es una herramienta de apoyo, no sustituye el criterio ni la responsabilidad académica del equipo.
## Umbral de decisión

Los puntajes continuos se convirtieron en decisiones mediante el costo centralizado `FN × Q4,200 + FP × Q180`. Cada umbral se eligió exclusivamente en VALIDATION evaluando todos los scores únicos; ante empates exactos se usa el umbral más alto. La simplificación representa daño esperado académico, no un ahorro individual garantizado de Q4,200 por cada verdadero positivo.

| Modelo | Threshold VALIDATION | FP | FN | Precision | Recall | F1 | Costo |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 0.055489 | 212 | 11 | 0.5058 | 0.9518 | 0.6606 | Q84,360 |
| B | 0.825399 | 406 | 41 | 0.3153 | 0.8202 | 0.4555 | Q245,280 |
| C | 0.668239 | 333 | 27 | 0.3764 | 0.8816 | 0.5276 | Q173,340 |

Antes de abrir TEST se congeló A con threshold `0.05548898134596948`: fue el mejor tanto en AP como en costo de VALIDATION. B probó que el orden contiene señal y C mejoró B, pero ninguno justificó desplazar a A. La evidencia está en `artefactos/final_decision_config.json`.

## Evaluación final

Los tres checkpoints se evaluaron una sola vez en TEST con sus thresholds congelados.

| Modelo | Test AP | Threshold | Precision | Recall | F1 | TP | FP | TN | FN | Costo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 0.9191 | 0.055489 | 0.5024 | 0.9505 | 0.6573 | 211 | 209 | 13,935 | 11 | Q83,820 |
| B | 0.7150 | 0.825399 | 0.3023 | 0.8784 | 0.4498 | 195 | 450 | 13,694 | 27 | Q194,400 |
| C | 0.8248 | 0.668239 | 0.3778 | 0.9054 | 0.5332 | 201 | 331 | 13,813 | 21 | Q147,780 |

## Impacto económico

En TEST, nunca bloquear cuesta Q932,400 y bloquear todo Q2,545,920. El candidato A cuesta Q83,820. Como el candidato final es el propio A, el ahorro frente a A es Q0; B y C agregarían Q110,580 y Q63,960 de costo, respectivamente. TEST cubre 26.8783 días: normalizando por 30.44 días, A cuesta Q94,927.01 por mes en la escala simulada. Esta cifra no debe extrapolarse directamente a 1.4 millones de tarjetas.

## Limitaciones observadas

A detectó 133/134 fraudes `testing_cashout` (99.25%), 71/75 `channel_takeover` (94.67%) y solo 7/13 `amount_anomaly` (53.85%). Seis de sus once falsos negativos pertenecen a este último mecanismo. Los falsos positivos incluyen microcompras, viajes, rachas de compras y una compra grande legítima; no encontramos un patrón único que explique todos. La recomendación preliminar es **CONSERVAR** A: las secuencias contienen información temporal real, pero en este experimento no mejoraron AP ni costo operativo frente al modelo agregado.

La recomendación cambiaría si una réplica temporal mostrara ahorro sostenido de B/C frente a A, si el costo o recall de un mecanismo crítico cruzara límites operativos acordados, si los falsos positivos fueran inaceptables, o si apareciera drift temporal. El modelado queda cerrado tras `artefactos/final_evaluation.json`.
