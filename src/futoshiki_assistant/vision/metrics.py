import cv2
import numpy as np

from .detection import order_points


def _distance(a, b):
    return float(
        np.linalg.norm(
            np.asarray(a, dtype=float)
            - np.asarray(b, dtype=float)
        )
    )


def analyze_capture_geometry(image, board_contour):
    height, width = image.shape[:2]
    points = order_points(board_contour)
    top_left, top_right, bottom_right, bottom_left = points

    top = _distance(top_left, top_right)
    bottom = _distance(bottom_left, bottom_right)
    left = _distance(top_left, bottom_left)
    right = _distance(top_right, bottom_right)

    horizontal_ratio = max(top, bottom) / max(
        min(top, bottom),
        1e-6,
    )

    vertical_ratio = max(left, right) / max(
        min(left, right),
        1e-6,
    )

    perspective_ratio = max(
        horizontal_ratio,
        vertical_ratio,
    )

    board_area_ratio = float(
        cv2.contourArea(
            points.astype(np.float32)
        )
    ) / float(height * width)

    min_x = float(points[:, 0].min())
    max_x = float(points[:, 0].max())
    min_y = float(points[:, 1].min())
    max_y = float(points[:, 1].max())

    margins = {
        "left": min_x / width,
        "right": (width - max_x) / width,
        "top": min_y / height,
        "bottom": (height - max_y) / height,
    }

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    return {
        "board_area_ratio": board_area_ratio,
        "min_margin_ratio": min(margins.values()),
        "perspective_ratio": perspective_ratio,
        "blur_score": float(
            cv2.Laplacian(
                gray,
                cv2.CV_64F,
            ).var()
        ),
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "margins": margins,
    }


def evaluate_operational_scope(
    metrics,
    source_type,
    settings,
):
    reasons = []
    warnings = []

    if source_type == "digital":
        if (
            metrics["perspective_ratio"]
            > settings.scope_max_perspective_ratio
        ):
            warnings.append(
                "Perspectiva digital superior al alcance base."
            )

        return {
            "within_scope": True,
            "reasons": reasons,
            "warnings": warnings,
        }

    board_area = metrics["board_area_ratio"]
    margin = metrics["min_margin_ratio"]
    perspective = metrics["perspective_ratio"]

    if board_area < settings.scope_min_board_area_ratio:
        reasons.append("El tablero ocupa muy poco de la imagen.")

    if board_area > settings.scope_max_board_area_ratio:
        reasons.append("El tablero ocupa demasiado de la imagen.")

    if margin < settings.scope_min_margin_ratio:
        reasons.append("El margen alrededor del tablero es insuficiente.")

    if perspective > settings.scope_max_perspective_ratio:
        reasons.append("La perspectiva supera el alcance base.")

    if metrics["blur_score"] < settings.scope_min_blur_score:
        warnings.append("La imagen presenta desenfoque significativo.")

    if metrics["brightness"] < settings.scope_min_brightness:
        warnings.append("La imagen presenta baja iluminación.")

    return {
        "within_scope": len(reasons) == 0,
        "reasons": reasons,
        "warnings": warnings,
    }
