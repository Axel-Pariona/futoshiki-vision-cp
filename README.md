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
data/evaluation/         Final evaluation dataset
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
pip install -e .
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

## Run one image

```bash
python -m futoshiki_assistant   --image path/to/image.jpg   --models-dir models   --source-type printed_photo   --output-dir results/example
```

For a clean digital input:

```bash
python -m futoshiki_assistant   --image path/to/board.png   --models-dir models   --source-type digital   --output-dir results/example
```

## Colab reproduction

Use:

`notebooks/colab_reproducible.ipynb`

The notebook clones the repository, installs dependencies, verifies the model artifacts, runs tests and processes an uploaded image end-to-end.

## Final dataset evaluation

Create `data/manifest.csv` from `data/manifest_template.csv`, then run:

```bash
python scripts/evaluate_dataset.py   --manifest data/manifest.csv   --models-dir models   --output results/dataset_results.csv
```

## Constraint Programming

For every cell, the base CSP creates:

`x[r,c] in {1, ..., N}`

and adds:

- `AllDifferent` for every row;
- `AllDifferent` for every column;
- equality constraints for given values;
- `<` or `>` constraints between adjacent cells.

`src/futoshiki_assistant/cp/joint.py` contains the advanced perception-aware model with boolean interpretation variables, `AddExactlyOne`, reified constraints and integer-scaled `-log(p)` costs.

## Final runtime configuration

All operational thresholds used for the final evaluation are centralized in:

`config/default.json`

The metadata stored with the V2 models is preserved separately under `models/`.

## Development artifacts

Old V5/V5.6 notebooks and ZIP kits are development history and are intentionally not required by the final package.

See `docs/GITHUB_AND_COLAB.md` for the clean GitHub/Colab workflow.
