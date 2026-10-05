# Final repository: `futoshiki-vision-cp`

# Code freeze

Versión base congelada para construir el dataset final.

## Alcance de visión

- 4x4
- 5x5
- imágenes digitales
- fotografías impresas
- perspectiva frontal o ligera
- tablero completo
- iluminación legible
- marco exterior visible en el formato principal de evaluación

## Parámetros congelados

- `digit_presence_threshold = 0.010`
- `inequality_min_confidence = 0.75`
- `inequality_margin_vs_blank = 0.20`
- `detection_score_fail = 0.70`
- `detection_score_warning = 0.85`
- `detection_coverage_fail = 0.80`
- `output_size = 900`

Los valores están centralizados en `config/default.json`.

## Regla experimental

El dataset final se crea después de este freeze.

Si una imagen del test final falla, el resultado se registra. No se modifica el código o los thresholds para hacer que esa misma imagen pase.

Si aparece una limitación grave que obliga a cambiar el sistema, se debe crear una nueva versión y repetir el benchmark completo con un nuevo split.
