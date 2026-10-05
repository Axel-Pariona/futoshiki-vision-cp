import argparse
import csv
import json
from pathlib import Path

from futoshiki_assistant.config import VisionSettings
from futoshiki_assistant.pipeline import FutoshikiPipeline
from futoshiki_assistant.vision.models import load_model_bundle


def main():
    parser = argparse.ArgumentParser(
        description="Evalúa el dataset final de Futoshiki.",
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

    manifest_path = Path(args.manifest)
    rows = list(
        csv.DictReader(
            manifest_path.open(
                encoding="utf-8"
            )
        )
    )

    results = []

    for row in rows:
        ground_truth = None

        if row.get("ground_truth"):
            ground_truth = json.loads(
                Path(
                    row["ground_truth"]
                ).read_text(
                    encoding="utf-8"
                )
            )

        try:
            result = pipeline.process(
                row["image"],
                source_type=row.get(
                    "source_type",
                    "digital",
                ),
                expected_n=(
                    int(row["size"])
                    if row.get("size")
                    else None
                ),
                ground_truth=ground_truth,
            )

            comparison = (
                result["comparison"] or {}
            )

            results.append(
                {
                    "id": row["id"],
                    "image": row["image"],
                    "size": result["size"],
                    "source_type": result["source_type"],
                    "detection_gate": result["detection_gate"],
                    "grid_score": result["grid_detection_score"],
                    "grid_coverage": result["grid_detection_coverage"],
                    "segmentation": result["segmentation_method"],
                    "givens_detected": len(
                        result["instance"]["givens"]
                    ),
                    "inequalities_detected": len(
                        result["instance"]["inequalities"]
                    ),
                    "cp_state": result["cp_state"],
                    "givens_exact": comparison.get("givens_exact"),
                    "inequalities_exact": comparison.get(
                        "inequalities_exact"
                    ),
                    "instance_exact": comparison.get("instance_exact"),
                    "total_time": result["total_time"],
                    "error": "",
                }
            )

        except Exception as error:
            results.append(
                {
                    "id": row["id"],
                    "image": row["image"],
                    "size": row.get("size", ""),
                    "source_type": row.get("source_type", ""),
                    "detection_gate": "FAIL",
                    "grid_score": "",
                    "grid_coverage": "",
                    "segmentation": "",
                    "givens_detected": "",
                    "inequalities_detected": "",
                    "cp_state": "",
                    "givens_exact": "",
                    "inequalities_exact": "",
                    "instance_exact": "",
                    "total_time": "",
                    "error": str(error),
                }
            )

    output = Path(args.output)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(results[0].keys())

    with output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)

    print(
        f"Resultados guardados en {output}"
    )


if __name__ == "__main__":
    main()
