# Protocolo del dataset final

El código y los thresholds se congelan antes de crear el conjunto de prueba final.

## Tamaños

- 4x4
- 5x5

## Categorías sugeridas

- digital limpia;
- impresa frontal;
- impresa con perspectiva ligera;
- iluminación moderada;
- mayor distancia manteniendo legibilidad;
- variante generada realista dentro del alcance.

## Ground truth

Cada puzzle lógico debe tener un JSON exacto con:

- tamaño;
- givens;
- desigualdades;
- estado lógico esperado.

Varias fotografías del mismo puzzle pueden compartir el mismo ground truth.

## Regla de evaluación

Las imágenes del conjunto de prueba no deben utilizarse para ajustar thresholds o reentrenar modelos.

Los casos que fallen deben conservarse y documentarse si pertenecen al alcance declarado.
