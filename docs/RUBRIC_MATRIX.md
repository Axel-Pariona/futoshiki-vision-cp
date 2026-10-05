# Matriz final de cumplimiento de la rúbrica

Esta matriz relaciona cada criterio con evidencia concreta del repositorio y del informe final.

| Criterio de rúbrica | Puntaje | Evidencia | Estado |
|---|---:|---|---|
| Precisión de grilla, números y símbolos | 3 | `vision/detection.py`, `vision/segmentation.py`, `vision/recognition.py`, benchmark final | Cubierto y medido |
| Diferentes condiciones de imagen | 1 | 10 casos main + 2 stress, `data/manifest.csv` | Cubierto |
| Formulación matemática CP | 3 | `cp/model.py`, informe final, variables/dominos/givens/desigualdades | Cubierto |
| Restricciones globales eficientes | 3 | `AddAllDifferent` por fila y columna | Cubierto |
| Restricciones reificadas | 1 | `cp/joint.py`, `AddExactlyOne`, `OnlyEnforceIf`, `reify_less_than` | Cubierto |
| Integración visión -> CP | 1 | `pipeline.py`: imagen -> instancia -> CP | Cubierto |
| Visualización clara | 1 | `vision/overlay.py`: segmentación, percepción y solución | Cubierto |
| Código limpio y modular | 2 | paquete `src/`, CLI, tests, configuración centralizada | Cubierto |
| Informe IEEE-like / LaTeX | 5 | informe final en LaTeX/PDF | Cubierto |

## Evidencia experimental final

Dataset:

- 10 imágenes main;
- 2 casos stress;
- 12 imágenes totales.

Resultados main:

- detección de tablero: 100%;
- givens exactos: 90%;
- desigualdades exactas: 70%;
- instancia completa exacta: 70%;
- estado CP correcto: 90%;
- éxito end-to-end: 70%.

Tiempos:

- chequeo de unicidad: 1.18 ms;
- resolución de una solución: 2.48 ms;
- CP total: 5.34 ms;
- pipeline total: 700.16 ms.

Tasas de error complementarias:

- givens: 10%;
- desigualdades: 30%;
- end-to-end: 30%.

## Constraint Programming

### Modelo base

Para cada celda:

`x[r,c] in {1, ..., N}`

Restricciones:

- `AllDifferent` en cada fila;
- `AllDifferent` en cada columna;
- `x[r,c] = value` para givens;
- `x[a] < x[b]` o `x[a] > x[b]` para desigualdades.

El modelo base es un CSP sin objetivo.

### Restricciones globales

Se usa `AddAllDifferent` directamente para filas y columnas, en lugar de descomponer la condición en desigualdades binarias por pares.

### Reificación

`reify_less_than()` implementa:

`B <-> (A < C)`

mediante:

- `A < C` si `B`;
- `A >= C` si `not B`.

El módulo probabilístico usa variables booleanas para seleccionar `<`, `>` o `blank`, junto con `AddExactlyOne`.

### COP probabilístico

La extensión avanzada transforma probabilidades visuales en costos enteros:

`cost = -log(p) * scale`

y minimiza la suma de los costos de las interpretaciones seleccionadas.

Esta extensión está implementada, pero no se mezcla con las métricas del benchmark determinista final.

## Integración y visualización

`FutoshikiPipeline.process()` realiza automáticamente:

```text
imagen
-> detección
-> rectificación
-> segmentación
-> reconocimiento
-> instancia JSON
-> CP-SAT
-> chequeo de unicidad
-> solución
-> overlays
```

El sistema produce vistas de:

- detección del tablero;
- tablero rectificado;
- segmentación;
- percepción;
- solución rectificada;
- solución proyectada sobre la imagen original.

## Reproducibilidad

El repositorio incluye:

- `requirements.txt`;
- `pyproject.toml`;
- checksums SHA-256 de modelos;
- ocho tests automatizados;
- notebook reproducible en Colab;
- dataset, manifest y ground truth;
- resultados finales congelados;
- código y configuración centralizados.

## Entregables

- código GitHub: completo;
- modelos: completos;
- dataset >= 10 imágenes: completo;
- ground truth: completo;
- benchmark y métricas: completos;
- análisis de errores: completo;
- tiempos del solver: completos;
- informe LaTeX/PDF: completo;
- demo/presentación de máximo 5 minutos: siguiente entregable a preparar.
