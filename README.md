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

Todavía no existen TRAIN, VALIDATION o TEST. El generador crea una única línea temporal completa y desconoce las futuras fronteras.

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
- **Evidencia que deberá revisarse posteriormente:** AUC-PR, costo, estabilidad y pruebas de falsificación en VALIDATION.
- **Veredicto final:** pendiente.

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
- **Evidencia que deberá revisarse posteriormente:** diferencia de AUC-PR e impacto económico en VALIDATION bajo el criterio predefinido.
- **Veredicto final:** pendiente.

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

Etapas 0 a 3 completadas. Quedan pendientes el preprocessing causal, el split estrictamente temporal y, posteriormente, el entrenamiento. El dataset permanece crudo: no hay ventanas agregadas, secuencias finales, normalización ni balanceo.

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
