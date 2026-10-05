# GitHub and Colab reproducibility

## Recommended repository

Name:

`futoshiki-vision-cp`

Description:

`End-to-end Futoshiki solver from images using computer vision and OR-Tools CP-SAT.`

## Files that belong in the repository

Upload the contents of this repository directly. Do not upload development ZIP files inside the repository.

The trained `.pt` files are included because they are small enough for regular Git hosting.

Development artifacts such as `Futoshiki_V5_Full_Kit.zip`, `Imagenes.zip` and old experimental notebooks should remain outside the final repository.

## Colab

Open:

`notebooks/colab_reproducible.ipynb`

Change only:

```python
GITHUB_USER = "your-github-user"
```

The notebook clones the repository, installs dependencies, verifies model hashes, runs tests and executes one uploaded image through the complete pipeline.

## Local reproduction

```bash
git clone https://github.com/YOUR_USER/futoshiki-vision-cp.git
cd futoshiki-vision-cp

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .

sha256sum -c models/checksums.sha256
pytest -q
```

Windows activation:

```powershell
.venv\Scripts\activate
```

## Final dataset

After the code freeze, add the curated evaluation images under:

`data/evaluation/images/`

and the exact JSON ground truth under:

`data/evaluation/ground_truth/`

Then create:

`data/manifest.csv`

and run:

```bash
python scripts/evaluate_dataset.py   --manifest data/manifest.csv   --models-dir models   --output results/dataset_results.csv
```
