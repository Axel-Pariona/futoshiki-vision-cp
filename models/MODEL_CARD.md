# Model card

## Artifacts

- `futoshiki_digit_model_v2.pt`: classifier for digits `1..6`.
- `futoshiki_inequality_model_v2.pt`: classifier for `blank`, `<`, `>`.
- `futoshiki_vision_v2_config.json`: metadata saved with the V2 training run.

## Final evaluation scope

The final computer-vision evaluation is limited to 4x4 and 5x5 boards.

The digit network still contains class `6` because the model was trained with a broader V2 class set. During inference, predictions outside `1..N` are rejected.

## Runtime thresholds

The metadata file contains the historical V2 presence threshold. The final runtime does not use that value as its operational threshold.

Final runtime parameters are defined only in:

`config/default.json`

Current values include:

- digit presence threshold: `0.010`
- inequality minimum confidence: `0.75`
- inequality margin over blank: `0.20`

This separation preserves the original model metadata while keeping the final evaluation configuration explicit and reproducible.

## Integrity

From the repository root, run:

```bash
sha256sum -c models/checksums.sha256
```

to verify that the model artifacts have not changed.
