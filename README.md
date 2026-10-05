# Futoshiki Vision + Constraint Programming

End-to-end system that interprets a Futoshiki board from an image and solves the extracted instance with OR-Tools CP-SAT.

## Final scope

Computer vision is evaluated on classical 4x4 and 5x5 boards under controlled or moderately variable conditions:

- digital images;
- photographs of printed boards;
- frontal or light perspective;
- complete board visible;
- sufficient illumination;
- visible outer frame in the main evaluation format.

The CP formulation remains parameterized for `N x N`.

## Pipeline

```text
Image
-> board detection and perspective correction
-> 4x4 / 5x5 size detection
-> adaptive ROI segmentation
-> digit-presence filtering
-> digit and inequality classification
-> canonical JSON instance
-> CP-SAT model
-> uniqueness check
-> solution and overlays
```

## Repository structure

```text
config/                  Final runtime thresholds
data/evaluation/         Final evaluation dataset and benchmark results
docs/                    Scope, rubric and reproducibility notes
models/                  Trained model artifacts
notebooks/               Minimal Colab reproduction notebook
scripts/                 Single-image and dataset evaluation
src/futoshiki_assistant/ Clean Python package
tests/                   Unit tests
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install .
```

Windows:

```powershell
.venv\Scripts\activate
```

## Verify model artifacts

```bash
sha256sum -c models/checksums.sha256
```

## Tests

```bash
pytest -q
```

The final repository contains eight automated tests covering the CP solver, uniqueness, reification, schema validation and visual presence scoring.

## Run one image

```bash
python -m futoshiki_assistant \
  --image path/to/image.jpg \
  --models-dir models \
  --source-type printed_photo \
  --output-dir results/example
```

For a clean digital input:

```bash
python -m futoshiki_assistant \
  --image path/to/board.png \
  --models-dir models \
  --source-type digital \
  --output-dir results/example
```

## Colab reproduction

Use:

`notebooks/colab_reproducible.ipynb`

The notebook clones the repository, installs dependencies, verifies model hashes, runs the automated tests and processes one uploaded image end-to-end.

## Final dataset

The final dataset contains **12 images**:

- **10 main-scope evaluation cases** used for the official metrics;
- **1 stress case** with colored/annotated inequality symbols;
- **1 out-of-scope stress case** containing handwritten completion values.

The vision evaluation is restricted to 4x4 and 5x5 boards. The logical CP model remains parameterized for `N x N`.

Dataset files:

```text
data/
  manifest.csv
  evaluation/
    images/
    ground_truth/
    results/
```

## Validate final ground truth

```bash
python scripts/validate_ground_truth.py
```

All ground-truth puzzles used in the benchmark are expected to return `UNIQUE`.

## Reproduce the final benchmark

```bash
python scripts/evaluate_dataset.py \
  --manifest data/manifest.csv \
  --models-dir models \
  --config config/default.json \
  --output results/dataset_results.csv
```

The frozen benchmark artifacts are versioned under:

```text
data/evaluation/results/
  dataset_results_final.csv
  summary_metrics_final.csv
  summary_metrics_final.json
```

## Final benchmark results

Official metrics are calculated only over the 10 cases with `scope_group = main`.

| Metric | Result |
|---|---:|
| Board detection success | 100% |
| Givens exact | 90% |
| Inequalities exact | 70% |
| Complete instance exact | 70% |
| CP status accuracy | 90% |
| End-to-end success | 70% |
| Mean CP time | 5.34 ms |
| Mean total pipeline time | 700.16 ms |

The corresponding error rates are:

- givens error rate: **10%**;
- inequalities error rate: **30%**;
- end-to-end error rate: **30%**.

Stress cases are reported separately and are not mixed with the main accuracy figures.

## Constraint Programming

For every cell, the base CSP creates:

`x[r,c] in {1, ..., N}`

and adds:

- `AllDifferent` for every row;
- `AllDifferent` for every column;
- equality constraints for given values;
- `<` or `>` constraints between adjacent cells.

The base model is a CSP without an objective function.

`src/futoshiki_assistant/cp/joint.py` contains the advanced perception-aware extension with boolean interpretation variables, `AddExactlyOne`, reified constraints and integer-scaled `-log(p)` costs. This extension is implemented but is not mixed with the official deterministic benchmark.

## Final runtime configuration

All operational thresholds used for the final evaluation are centralized in:

`config/default.json`

The final benchmark was executed after freezing these parameters. Final test images were not used to retune thresholds or retrain the models.

## Reproducibility

The project provides:

- source code and trained model artifacts;
- SHA-256 checksums for the model files;
- `requirements.txt` and `pyproject.toml`;
- automated tests;
- a reproducible Colab notebook;
- final dataset manifest and ground truth;
- frozen benchmark CSV/JSON results.

## Code and report

Repository:

`https://github.com/Axel-Pariona/futoshiki-vision-cp`

The final report documents the computer-vision pipeline, the formal CP model, global and reified constraints, integration, experimental protocol, benchmark results, complexity and solver timings.
