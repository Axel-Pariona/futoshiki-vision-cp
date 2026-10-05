# Final dataset protocol

The code and runtime thresholds were frozen before the final benchmark.

## Composition

The repository contains **12 images**:

- **10 main-scope cases**;
- **1 colored/annotated-symbol stress case**;
- **1 handwritten out-of-scope stress case**.

Primary accuracy metrics use only:

`scope_group = main`

Stress cases are reported separately.

## Sizes

- 4x4
- 5x5

The vision subsystem is evaluated only for these sizes. The CP model is parameterized for general `N x N`.

## Acquisition and visual variation

The collection includes:

- clean digital boards;
- colored digital layouts;
- high-contrast digital layouts;
- real printed photographs;
- lower-contrast photography;
- mild perspective;
- generated photo-like variants;
- alternative editorial layouts;
- dense inequality layouts;
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

Different visual captures of the same logical puzzle may share the same ground truth.

## Final evaluation rule

Do not modify thresholds or retrain models after inspecting final test results.

If an in-scope image fails, the failure remains part of the benchmark.

Ground truth is validated independently before the visual benchmark:

```bash
python scripts/validate_ground_truth.py
```

The final dataset benchmark is reproduced with:

```bash
python scripts/evaluate_dataset.py \
  --manifest data/manifest.csv \
  --models-dir models \
  --config config/default.json \
  --output results/dataset_results.csv
```

## Frozen benchmark artifacts

The final results are versioned under:

`data/evaluation/results/`

Files:

- `dataset_results_final.csv`
- `summary_metrics_final.csv`
- `summary_metrics_final.json`

Official main-scope results:

- board detection success: 100%;
- givens exact: 90%;
- inequalities exact: 70%;
- instance exact: 70%;
- CP status accuracy: 90%;
- end-to-end success: 70%;
- mean CP time: 5.34 ms;
- mean total pipeline time: 700.16 ms.
