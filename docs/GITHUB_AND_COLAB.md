# GitHub and Colab reproducibility

## Repository

`https://github.com/Axel-Pariona/futoshiki-vision-cp`

Description:

`End-to-end Futoshiki solver from images using computer vision and OR-Tools CP-SAT.`

## Repository contents

The final repository contains:

- clean Python source code;
- trained model artifacts;
- runtime configuration;
- automated tests;
- reproducible Colab notebook;
- final dataset and ground truth;
- frozen benchmark results;
- documentation.

Development ZIP files and obsolete experimental notebooks are intentionally excluded from the final workflow.

## Colab

Use:

`notebooks/colab_reproducible.ipynb`

Set:

```python
GITHUB_USER = "Axel-Pariona"
REPO_NAME = "futoshiki-vision-cp"
BRANCH = "main"
```

The notebook:

1. clones the repository;
2. installs dependencies;
3. verifies model hashes;
4. runs the automated tests;
5. lets the user upload one 4x4 or 5x5 image;
6. executes the complete end-to-end pipeline;
7. shows detection, rectification, segmentation, perception and final solution overlays.

## Local reproduction

```bash
git clone https://github.com/Axel-Pariona/futoshiki-vision-cp.git
cd futoshiki-vision-cp

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install .

sha256sum -c models/checksums.sha256
pytest -q
```

Windows activation:

```powershell
.venv\Scripts\activate
```

## Ground-truth validation

```bash
python scripts/validate_ground_truth.py
```

## Final benchmark

```bash
python scripts/evaluate_dataset.py \
  --manifest data/manifest.csv \
  --models-dir models \
  --config config/default.json \
  --output results/dataset_results.csv
```

Frozen benchmark results are stored in:

`data/evaluation/results/`

## Final dataset composition

- 10 main-scope cases;
- 1 stress case;
- 1 out-of-scope stress case;
- 12 images total.

## Final benchmark metrics

Main-scope results:

- board detection success: 100%;
- givens exact: 90%;
- inequalities exact: 70%;
- complete instance exact: 70%;
- CP status accuracy: 90%;
- end-to-end success: 70%.

Mean times:

- CP total: 5.34 ms;
- complete pipeline: 700.16 ms.

## Experimental policy

The final benchmark is frozen.

Final test images must not be used to:

- retune thresholds;
- retrain the models;
- remove failed cases;
- alter ground truth.

Any future modification requires a new experimental version and a complete rerun.
