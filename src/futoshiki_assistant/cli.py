import argparse
import json
from pathlib import Path

import cv2

from .config import VisionSettings
from .pipeline import FutoshikiPipeline
from .vision.models import load_model_bundle


def _save_image(path, image):
    if image is not None:
        cv2.imwrite(str(path), image)


def main():
    parser = argparse.ArgumentParser(
        description="Futoshiki Assistant 4x4/5x5",
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Ruta de la imagen.",
    )

    parser.add_argument(
        "--models-dir",
        default="models",
        help="Directorio con los modelos V2.",
    )

    parser.add_argument(
        "--config",
        default="config/default.json",
        help="Archivo JSON de configuración.",
    )

    parser.add_argument(
        "--source-type",
        default="digital",
        choices=[
            "digital",
            "printed_photo",
            "screen_photo",
            "generated_photo",
        ],
    )

    parser.add_argument(
        "--expected-n",
        type=int,
        choices=[4, 5],
        default=None,
    )

    parser.add_argument(
        "--ground-truth",
        default=None,
        help="JSON opcional de ground truth.",
    )

    parser.add_argument(
        "--output-dir",
        default="results",
    )

    args = parser.parse_args()

    settings = VisionSettings.from_json(args.config)
    model_bundle = load_model_bundle(args.models_dir)

    ground_truth = None

    if args.ground_truth:
        ground_truth = json.loads(
            Path(args.ground_truth).read_text(
                encoding="utf-8"
            )
        )

    pipeline = FutoshikiPipeline(
        model_bundle,
        settings,
    )

    result = pipeline.process(
        args.image,
        source_type=args.source_type,
        expected_n=args.expected_n,
        ground_truth=ground_truth,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = {
        key: value
        for key, value in result.items()
        if key not in {
            "details",
            "board_overlay",
            "rectified",
            "segmentation_overlay",
            "perception_overlay",
            "solution_overlay",
            "final_overlay",
        }
    }

    (
        output_dir
        / "result.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    _save_image(
        output_dir / "board_detection.png",
        result["board_overlay"],
    )

    _save_image(
        output_dir / "rectified.png",
        result["rectified"],
    )

    _save_image(
        output_dir / "segmentation.png",
        result["segmentation_overlay"],
    )

    _save_image(
        output_dir / "perception.png",
        result["perception_overlay"],
    )

    _save_image(
        output_dir / "solution_rectified.png",
        result["solution_overlay"],
    )

    _save_image(
        output_dir / "solution_original.png",
        result["final_overlay"],
    )

    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
