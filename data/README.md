# Evaluation data

The final dataset is added only after the code freeze.

## Final structure

```text
data/
  evaluation/
    images/
    ground_truth/
  manifest.csv
```

Each image in `evaluation/images/` must reference the ground-truth JSON of the logical puzzle that it represents.

Different photographs of the same puzzle may share one ground-truth file.

Do not tune thresholds or retrain models using the final test images.
