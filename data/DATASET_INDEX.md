# Final evaluation dataset

The repository contains **11 evaluation images**:

- **9 main-scope cases** used for the primary evaluation.
- **1 stress case** with colored/annotated inequality symbols.
- **1 out-of-scope stress case** containing handwritten completion values.

All images have an explicit ground-truth JSON. Several photographs may share the same logical puzzle.

## Images

| ID | Size | Type | Condition | Group | Puzzle |
|---|---:|---|---|---|---|
| eval_01 | 5x5 | digital | digital_clean | main | puzzle_5x5_A |
| eval_02 | 5x5 | digital | digital_colored_background | main | puzzle_5x5_B |
| eval_03 | 5x5 | digital | digital_dark_frame | main | puzzle_5x5_C |
| eval_04 | 4x4 | printed_photo | real_photo_green_light_angle | main | puzzle_4x4_A |
| eval_05 | 5x5 | printed_photo | real_photo_low_contrast | main | puzzle_5x5_A |
| eval_06 | 4x4 | generated_photo | generated_photo_light_angle | main | puzzle_4x4_B |
| eval_07 | 5x5 | generated_photo | generated_photo_light_angle | main | puzzle_5x5_D |
| eval_08 | 4x4 | printed_photo | annotated_colored_symbols | stress | puzzle_4x4_D |
| eval_09 | 5x5 | digital | digital_book_cover | main | puzzle_5x5_E |
| eval_10 | 4x4 | digital | digital_dense_inequalities | main | puzzle_4x4_C |
| eval_11 | 5x5 | digital | handwritten_completion | stress_out_of_scope | puzzle_5x5_F |

## Experimental policy

Primary metrics should be reported for `scope_group = main`.

Stress cases should be reported separately and used to discuss limitations. They should not be mixed with the main accuracy figure without clearly labeling the difference.

`eval_01` and `eval_05` represent the same logical puzzle under different acquisition conditions, allowing a direct comparison between clean digital input and a real photograph.

The dataset was organized after code freeze. Final test images must not be used to retune thresholds or retrain the models.
