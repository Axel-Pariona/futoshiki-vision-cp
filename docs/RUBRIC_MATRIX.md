# Matriz de cumplimiento de la rúbrica

Esta matriz separa lo que ya está cerrado en código de la evidencia que todavía depende del dataset, el informe y la presentación.

| Criterio de rúbrica | Puntaje | Evidencia preparada | Estado |
|---|---:|---|---|
| Pipeline de visión computacional | 3 | `vision/detection.py`, `vision/segmentation.py`, `vision/metrics.py` | Código cerrado |
| Reconocimiento visual | 1 | `vision/models.py`, `vision/recognition.py`, gates de dígitos y desigualdades | Código cerrado; benchmark pendiente |
| Formulación CP | 3 | `cp/model.py`, variables `1..N`, givens y desigualdades | Cerrado |
| Restricciones globales eficientes | 3 | `AddAllDifferent` por fila y columna | Cerrado |
| Restricciones reificadas | 1 | `cp/joint.py`, `AddExactlyOne`, `OnlyEnforceIf`, `reify_less_than` | Cerrado |
| Integración visión a CP | 1 | `pipeline.py`: imagen -> instancia -> CP | Cerrado |
| Visualización de resultado | 1 | `vision/overlay.py`: segmentación, percepción y solución | Cerrado |
| Código limpio y modular | 2 | paquete `src/`, CLI, tests, configuración centralizada | Cerrado |
| Informe IEEE-like / LaTeX | 5 | se redactará después del benchmark | Pendiente |

Total asociado directamente al código antes del informe: **15/15 puntos técnicamente cubiertos**.

Los 5 puntos del informe todavía no pueden darse por cerrados porque requieren redactar y presentar la evidencia experimental.

## Entregables obligatorios que todavía faltan

Además de la rúbrica de código/modelado, el trabajo necesita evidencia experimental y entregables:

- dataset final con al menos 10 imágenes;
- condiciones visuales variadas;
- ground truth por puzzle lógico;
- métricas de visión;
- tiempos de solver;
- análisis de errores;
- README y requirements reproducibles;
- informe final;
- demo aproximada de cinco minutos.

README, requirements, CLI y scripts de evaluación ya están preparados en este repositorio. El dataset y sus resultados todavía deben construirse.

## Evidencia de Constraint Programming

El informe debe explicar explícitamente:

### Modelo base

Para cada celda:

`x[r,c] in {1, ..., N}`

Restricciones:

- `AllDifferent` en cada fila;
- `AllDifferent` en cada columna;
- `x[r,c] = value` para givens;
- `x[a] < x[b]` o `x[a] > x[b]` para desigualdades.

El modelo base es un CSP, no un COP.

### Global constraints

`AllDifferent` se usa directamente en vez de descomponer filas y columnas en múltiples desigualdades binarias.

### Reificación

`reify_less_than()` implementa equivalencia completa:

`B <-> (A < C)`

mediante:

- `A < C` si `B`;
- `A >= C` si `not B`.

El modelo probabilístico usa variables booleanas para seleccionar `<`, `>` o `blank`, junto con `AddExactlyOne`.

### COP probabilístico

El modo avanzado convierte probabilidades visuales en costos enteros:

`cost = -log(p) * scale`

y minimiza la suma de los costos de las interpretaciones seleccionadas.

## Evidencia que debe producir el dataset final

Para cada imagen se debe registrar al menos:

- detección correcta del tablero;
- tamaño 4x4 o 5x5;
- givens detectados;
- desigualdades detectadas;
- `givens_exact`;
- `inequalities_exact`;
- `instance_exact`;
- estado CP;
- éxito end-to-end;
- tiempo total;
- tiempo de CP;
- condición de captura;
- observaciones de error.

## Regla de congelamiento

A partir de esta versión:

- no se ajustan thresholds usando el test final;
- no se reentrenan modelos con imágenes del test final;
- cualquier ajuste futuro requiere separar development y test;
- los fallos dentro del alcance se conservan en los resultados.
