from pathlib import Path
import time

import cv2

from .config import VisionSettings
from .cp.solver import check_uniqueness, solve_futoshiki
from .schema import compare_instances
from .vision.detection import (
    detect_and_rectify_board,
    draw_board_detection,
)
from .vision.metrics import (
    analyze_capture_geometry,
    evaluate_operational_scope,
)
from .vision.overlay import (
    draw_perception_overlay,
    draw_segmentation_overlay,
    draw_solution_rectified,
    project_overlay_to_original,
)
from .vision.recognition import recognize_board
from .vision.segmentation import segment_rectified_board


class FutoshikiPipeline:
    def __init__(
        self,
        model_bundle,
        settings=None,
    ):
        self.model_bundle = model_bundle
        self.settings = settings or VisionSettings()

    def _detection_gate(
        self,
        score,
        coverage,
    ):
        if coverage < self.settings.detection_coverage_fail:
            return "FAIL"

        if score < self.settings.detection_score_fail:
            return "FAIL"

        if score < self.settings.detection_score_warning:
            return "WARNING"

        return "PASS"

    def process(
        self,
        image_path,
        source_type="digital",
        expected_n=None,
        ground_truth=None,
    ):
        image_path = Path(image_path)
        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError(
                f"No se pudo abrir la imagen: {image_path}"
            )

        start = time.perf_counter()

        detection = detect_and_rectify_board(
            image,
            expected_n=expected_n,
            output_size=self.settings.output_size,
        )

        n = int(detection["detected_n"])

        if n not in self.settings.supported_sizes:
            raise ValueError(
                f"El tamaño detectado {n} no está soportado."
            )

        detection_score = float(
            detection["grid_detection_score"]
        )

        detection_coverage = float(
            detection["grid_info"].get(
                "coverage",
                0.0,
            )
        )

        gate = self._detection_gate(
            detection_score,
            detection_coverage,
        )

        if gate == "FAIL":
            raise RuntimeError(
                "FAIL_BOARD_DETECTION: "
                f"score={detection_score:.3f}, "
                f"coverage={detection_coverage:.3f}"
            )

        geometry = analyze_capture_geometry(
            image,
            detection["board_contour"],
        )

        scope = evaluate_operational_scope(
            geometry,
            source_type,
            self.settings,
        )

        segmentation = segment_rectified_board(
            detection["rectified"],
            n,
            mode=self.settings.segmentation_mode,
        )

        recognition = recognize_board(
            segmentation,
            n,
            self.model_bundle,
            digit_presence_threshold=(
                self.settings.digit_presence_threshold
            ),
            inequality_min_confidence=(
                self.settings.inequality_min_confidence
            ),
            inequality_margin_vs_blank=(
                self.settings.inequality_margin_vs_blank
            ),
        )

        instance = recognition["instance"]
        details = recognition["details"]

        cp_start = time.perf_counter()
        uniqueness = check_uniqueness(instance)
        solution_result = solve_futoshiki(instance)
        cp_time = time.perf_counter() - cp_start

        elapsed = time.perf_counter() - start

        segmentation_overlay = draw_segmentation_overlay(
            detection["rectified"],
            segmentation["roi_boxes"],
            n,
        )

        perception_overlay = draw_perception_overlay(
            detection["rectified"],
            segmentation["roi_boxes"],
            details,
            n,
        )

        solution_overlay = None
        final_overlay = None

        if solution_result["solution"] is not None:
            solution_overlay = draw_solution_rectified(
                detection["rectified"],
                solution_result["solution"],
                instance["givens"],
                n,
            )

            final_overlay = project_overlay_to_original(
                detection,
                solution_overlay,
            )

        comparison = (
            compare_instances(
                ground_truth,
                instance,
            )
            if ground_truth is not None
            else None
        )

        return {
            "image_path": str(image_path),
            "size": n,
            "source_type": source_type,
            "detection_method": detection["detector_method"],
            "detection_gate": gate,
            "grid_detection_score": detection_score,
            "grid_detection_coverage": detection_coverage,
            "geometry": geometry,
            "scope": scope,
            "segmentation_method": segmentation["method"],
            "instance": instance,
            "details": details,
            "cp_state": uniqueness["status"],
            "uniqueness_time": uniqueness["time"],
            "solve_time": solution_result["solve_time"],
            "cp_time": cp_time,
            "solution": solution_result["solution"],
            "comparison": comparison,
            "total_time": elapsed,
            "board_overlay": draw_board_detection(
                image,
                detection,
            ),
            "rectified": detection["rectified"],
            "segmentation_overlay": segmentation_overlay,
            "perception_overlay": perception_overlay,
            "solution_overlay": solution_overlay,
            "final_overlay": final_overlay,
        }
