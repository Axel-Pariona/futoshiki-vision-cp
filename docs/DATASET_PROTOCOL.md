# Final dataset protocol

The code and runtime thresholds are frozen before evaluating these images.

## Composition

The repository contains 11 images:

- 9 main-scope cases;
- 1 colored/annotated-symbol stress case;
- 1 handwritten out-of-scope stress case.

The primary accuracy metrics must use `scope_group = main`.

Stress cases are reported separately to document limitations.

## Sizes

- 4x4
- 5x5

## Acquisition and visual variation

The collection includes:

- clean digital boards;
- colored digital layouts;
- high-contrast digital layouts;
- real printed photographs;
- lower-contrast photography;
- mild perspective;
- generated photo-like variants;
- alternative editorial layout;
- dense inequality layout;
- annotated and handwritten stress cases.

## Ground truth

Every image references a JSON file under:

`data/evaluation/ground_truth/`

Each JSON contains:

- board size;
- given values;
- inequalities;
- expected CP status;
- verified solution.

Different visual captures of the same logical puzzle share the same ground truth.

## Evaluation rule

Do not modify thresholds or retrain models after inspecting final test results.

If an in-scope image fails, the failure remains part of the benchmark.

Run:

```bash
python scripts/validate_ground_truth.py
python scripts/evaluate_dataset.py   --manifest data/manifest.csv   --models-dir models   --output results/dataset_results.csv
```
