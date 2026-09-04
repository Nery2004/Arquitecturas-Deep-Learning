"""Evaluación económica futura de falsos negativos y falsos positivos.

El costo acordado es Q4,200 por fraude no detectado y Q180 por transacción
legítima bloqueada. El threshold se seleccionará solo con VALIDATION.
"""

FALSE_NEGATIVE_COST_GTQ: int = 4_200
FALSE_POSITIVE_COST_GTQ: int = 180
