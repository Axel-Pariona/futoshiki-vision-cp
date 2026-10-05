import argparse
import csv
import json
from pathlib import Path

from futoshiki_assistant.config import VisionSettings
from futoshiki_assistant.pipeline import FutoshikiPipeline
from futoshiki_assistant.vision.models import load_model_bundle


def parse_optional_int(value):
    return int(value) if value else None


def parse_optional_json(path):
    if not path:
        return None

    return json.loads(
        Path(path).read_text(
            encoding="utf-8"
        )
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the final Futoshiki image dataset."
    )

    parser.add_argument(
        "--manifest",
        required=True,
    )

    parser.add_argument(
        "--models-dir",
        default="models",
    )

    parser.add_argument(
        "--config",
        default="config/default.json",
    )

    parser.add_argument(
        "--output",
        default="results/dataset_results.csv",
    )

    args = parser.parse_args()

    settings = VisionSettings.from_json(
        args.config
    )

    models = load_model_bundle(
        args.models_dir
    )

    pipeline = FutoshikiPipeline(
        models,
        settings,
    )

    with Path(args.manifest).open(
        encoding="utf-8"
    ) as handle:
        rows = list(
            csv.DictReader(handle)
        )

    results = []

    for row in rows:
        ground_truth = parse_optional_json(
            row.get("ground_truth")
        )

        expected_status = row.get(
            "expected_status",
            "",
        )

        try:
            result = pipeline.process(
                row["image"],
                source_type=row.get(
                    "source_type",
                    "digital",
                ),
                expected_n=parse_optional_int(
                    row.get("size")
                ),
                ground_truth=ground_truth,
            )

            comparison = (
                result["comparison"]
                or {}
            )

            cp_matches_expected = (
                result["cp_state"]
                == expected_status
            ) if expected_status else None

            end_to_end_success = (
                comparison.get(
                    "instance_exact"
                ) is True
                and
                cp_matches_expected is True
            )

            results.append(
                {
                    "id": row["id"],
                    "puzzle_id": row.get("puzzle_id", ""),
                    "image": row["image"],
                    "size": result["size"],
                    "source_type": result["source_type"],
                    "condition": row.get("condition", ""),
                    "scope_group": row.get("scope_group", ""),
                    "detection_gate": result["detection_gate"],
                    "grid_score": result["grid_detection_score"],
                    "grid_coverage": result["grid_detection_coverage"],
                    "segmentation": result["segmentation_method"],
                    "givens_expected": (
                        len(ground_truth.get("givens", []))
                        if ground_truth
                        else ""
                    ),
                    "givens_detected": len(
                        result["instance"]["givens"]
                    ),
                    "inequalities_expected": (
                        len(
                            ground_truth.get(
                                "inequalities",
                                [],
                            )
                        )
                        if ground_truth
                        else ""
                    ),
                    "inequalities_detected": len(
                        result["instance"]["inequalities"]
                    ),
                    "givens_exact": comparison.get("givens_exact"),
                    "inequalities_exact": comparison.get(
                        "inequalities_exact"
                    ),
                    "instance_exact": comparison.get("instance_exact"),
                    "expected_status": expected_status,
                    "cp_state": result["cp_state"],
                    "cp_matches_expected": cp_matches_expected,
                    "end_to_end_success": end_to_end_success,
                    "uniqueness_time_s": result["uniqueness_time"],
                    "solve_time_s": result["solve_time"],
                    "cp_time_s": result["cp_time"],
                    "total_time_s": result["total_time"],
                    "error": "",
                }
            )

        except Exception as error:
            results.append(
                {
                    "id": row["id"],
                    "puzzle_id": row.get("puzzle_id", ""),
                    "image": row["image"],
                    "size": row.get("size", ""),
                    "source_type": row.get("source_type", ""),
                    "condition": row.get("condition", ""),
                    "scope_group": row.get("scope_group", ""),
                    "detection_gate": "FAIL",
                    "grid_score": "",
                    "grid_coverage": "",
                    "segmentation": "",
                    "givens_expected": "",
                    "givens_detected": "",
                    "inequalities_expected": "",
                    "inequalities_detected": "",
                    "givens_exact": "",
                    "inequalities_exact": "",
                    "instance_exact": "",
                    "expected_status": expected_status,
                    "cp_state": "",
                    "cp_matches_expected": "",
                    "end_to_end_success": False,
                    "uniqueness_time_s": "",
                    "solve_time_s": "",
                    "cp_time_s": "",
                    "total_time_s": "",
                    "error": str(error),
                }
            )

    output = Path(args.output)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                results[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(results)

    print(
        f"Results saved to {output}"
    )


if __name__ == "__main__":
    main()
