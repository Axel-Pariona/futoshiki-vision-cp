# Final repository: `futoshiki-vision-cp`

# Code, dataset and results freeze

This document records the configuration used for the final benchmark and report.

## Vision scope

- 4x4
- 5x5
- digital images
- printed photographs
- frontal or light perspective
- complete board visible
- sufficient illumination
- visible outer frame in the main evaluation format

The CP model remains parameterized for `N x N`.

## Frozen runtime parameters

- `digit_presence_threshold = 0.010`
- `inequality_min_confidence = 0.75`
- `inequality_margin_vs_blank = 0.20`
- `detection_score_fail = 0.70`
- `detection_score_warning = 0.85`
- `detection_coverage_fail = 0.80`
- `output_size = 900`

All values are centralized in:

`config/default.json`

## Freeze policy

The final test dataset was evaluated after freezing the code and runtime thresholds.

After inspecting the final benchmark:

- thresholds were not retuned;
- models were not retrained;
- failed in-scope images were not removed;
- ground truth was not altered to match predictions.

Any future model or threshold change must create a new experimental version and rerun the full benchmark.

## Frozen dataset

The final dataset contains:

- 10 main-scope cases;
- 1 stress case;
- 1 out-of-scope stress case;
- 12 images in total.

Manifest:

`data/manifest.csv`

## Frozen results

Final benchmark files:

```text
data/evaluation/results/
  dataset_results_final.csv
  summary_metrics_final.csv
  summary_metrics_final.json
```

Main-scope metrics:

- board detection success: 100%;
- givens exact: 90%;
- inequalities exact: 70%;
- complete instance exact: 70%;
- CP status accuracy: 90%;
- end-to-end success: 70%;
- mean CP time: 5.34 ms;
- mean total pipeline time: 700.16 ms.

These are the values used in the final report.
