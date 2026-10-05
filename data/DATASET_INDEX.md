# Final evaluation dataset

The repository contains **12 evaluation images**:

- **10 main-scope cases** used for the primary evaluation.
- **1 stress case** with colored/annotated inequality symbols.
- **1 out-of-scope stress case** containing handwritten completion values.

All images have an explicit ground-truth JSON. Different visual captures may share the same logical puzzle.

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
| eval_12 | 4x4 | printed_photo | real_photo_frontal_clean | main | puzzle_4x4_A |

## Main benchmark

Primary metrics are calculated only for:

`scope_group = main`

This gives **10 official evaluation images**.

Frozen results:

| Metric | Result |
|---|---:|
| Board detection success | 100% |
| Givens exact | 90% |
| Inequalities exact | 70% |
| Instance exact | 70% |
| CP status accuracy | 90% |
| End-to-end success | 70% |

## Stress cases

`eval_08` and `eval_11` are reported separately to document behavior outside or at the edge of the declared scope.

They are not included in the main accuracy figures.

## Shared logical puzzles

`eval_01` and `eval_05` represent the same logical 5x5 puzzle under different acquisition conditions.

`eval_04` and `eval_12` represent the same logical 4x4 puzzle under different acquisition conditions.

This makes it possible to compare perception performance while keeping the underlying CP instance fixed.

## Experimental policy

The dataset was organized after code freeze.

Final test images were not used to:

- retune thresholds;
- retrain the models;
- replace failed samples.

Failures inside the declared scope remain part of the benchmark.
