import math
from typing import Optional

import cv2
import numpy as np


SUPPORTED_N = (4, 5)


def order_points(points):
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)

    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)

    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]      # TL
    ordered[2] = pts[np.argmax(s)]      # BR
    ordered[1] = pts[np.argmin(d)]      # TR
    ordered[3] = pts[np.argmax(d)]      # BL

    return ordered


def quad_area(quad):
    q = order_points(quad)
    return float(cv2.contourArea(q.astype(np.float32)))


def warp_quad(image, quad, output_size=900):
    src = order_points(quad)

    dst = np.array(
        [
            [0, 0],
            [output_size - 1, 0],
            [output_size - 1, output_size - 1],
            [0, output_size - 1],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(src, dst)
    inverse_matrix = cv2.getPerspectiveTransform(dst, src)

    rectified = cv2.warpPerspective(
        image,
        matrix,
        (output_size, output_size),
        flags=cv2.INTER_CUBIC,
    )

    return {
        "original": image,
        "rectified": rectified,
        "board_contour": src.reshape(-1, 1, 2).astype(np.int32),
        "quad": src,
        "matrix": matrix,
        "inverse_matrix": inverse_matrix,
    }


def _segment_angle_degrees(segment):
    x1, y1, x2, y2 = map(float, segment)
    angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1))) % 180.0

    if angle > 90.0:
        angle = 180.0 - angle

    return angle


def _fit_horizontal_line(segments):
    points = []

    for x1, y1, x2, y2 in segments:
        points.extend([(x1, y1), (x2, y2)])

    points = np.asarray(points, dtype=np.float64)

    if len(points) < 4:
        return None

    m, b = np.polyfit(points[:, 0], points[:, 1], 1)

    # y = m*x + b -> m*x - y + b = 0
    line = np.array([m, -1.0, b], dtype=np.float64)
    line /= max(np.linalg.norm(line[:2]), 1e-9)

    return line


def _fit_vertical_line(segments):
    points = []

    for x1, y1, x2, y2 in segments:
        points.extend([(x1, y1), (x2, y2)])

    points = np.asarray(points, dtype=np.float64)

    if len(points) < 4:
        return None

    # x = m*y + b -> x - m*y - b = 0
    m, b = np.polyfit(points[:, 1], points[:, 0], 1)

    line = np.array([1.0, -m, -b], dtype=np.float64)
    line /= max(np.linalg.norm(line[:2]), 1e-9)

    return line


def _line_intersection(line1, line2):
    p = np.cross(line1, line2)

    if abs(p[2]) < 1e-9:
        return None

    return p[:2] / p[2]


def _two_cluster_extremes(values):
    values = np.asarray(values, dtype=np.float32).reshape(-1, 1)

    if len(values) < 2:
        return None

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        50,
        0.1,
    )

    _, labels, centers = cv2.kmeans(
        values,
        2,
        None,
        criteria,
        10,
        cv2.KMEANS_PP_CENTERS,
    )

    centers = centers.flatten()
    order = np.argsort(centers)

    low_label = int(order[0])
    high_label = int(order[1])

    return labels.flatten(), low_label, high_label, centers


def detect_blue_frame_hough(image):
    """
    Detector prioritario para el formato canónico final.

    Busca un marco azul oscuro alrededor del tablero.
    Es deliberadamente específico: la estandarización del formato
    aumenta muchísimo la estabilidad frente a perspectiva.
    """

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Rango amplio de azules para impresión/fotografía.
    lower = np.array([88, 25, 25], dtype=np.uint8)
    upper = np.array([135, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)

    # Eliminar píxeles aislados y unir segmentos del marco.
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones((2, 2), np.uint8),
        iterations=1,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), np.uint8),
        iterations=2,
    )

    h, w = mask.shape
    min_dim = min(h, w)

    lines = cv2.HoughLinesP(
        mask,
        rho=1,
        theta=np.pi / 180.0,
        threshold=max(45, int(min_dim * 0.045)),
        minLineLength=max(80, int(min_dim * 0.20)),
        maxLineGap=max(20, int(min_dim * 0.07)),
    )

    if lines is None:
        return None

    horizontal = []
    vertical = []

    for line in lines[:, 0, :]:
        angle = _segment_angle_degrees(line)

        if angle <= 30.0:
            horizontal.append(tuple(map(int, line)))

        elif angle >= 60.0:
            vertical.append(tuple(map(int, line)))

    if len(horizontal) < 2 or len(vertical) < 2:
        return None

    h_mid = [
        (segment[1] + segment[3]) / 2.0
        for segment in horizontal
    ]

    v_mid = [
        (segment[0] + segment[2]) / 2.0
        for segment in vertical
    ]

    h_cluster = _two_cluster_extremes(h_mid)
    v_cluster = _two_cluster_extremes(v_mid)

    if h_cluster is None or v_cluster is None:
        return None

    h_labels, h_low, h_high, _ = h_cluster
    v_labels, v_low, v_high, _ = v_cluster

    top_segments = [
        segment
        for segment, label in zip(horizontal, h_labels)
        if int(label) == h_low
    ]

    bottom_segments = [
        segment
        for segment, label in zip(horizontal, h_labels)
        if int(label) == h_high
    ]

    left_segments = [
        segment
        for segment, label in zip(vertical, v_labels)
        if int(label) == v_low
    ]

    right_segments = [
        segment
        for segment, label in zip(vertical, v_labels)
        if int(label) == v_high
    ]

    top = _fit_horizontal_line(top_segments)
    bottom = _fit_horizontal_line(bottom_segments)
    left = _fit_vertical_line(left_segments)
    right = _fit_vertical_line(right_segments)

    if any(item is None for item in [top, bottom, left, right]):
        return None

    corners = [
        _line_intersection(top, left),
        _line_intersection(top, right),
        _line_intersection(bottom, right),
        _line_intersection(bottom, left),
    ]

    if any(corner is None for corner in corners):
        return None

    quad = order_points(
        np.asarray(corners, dtype=np.float32)
    )

    # Validación básica.
    area_ratio = quad_area(quad) / float(h * w)

    if not (0.08 <= area_ratio <= 0.98):
        return None

    margin_x = 0.18 * w
    margin_y = 0.18 * h

    if (
        quad[:, 0].min() < -margin_x
        or quad[:, 0].max() > w + margin_x
        or quad[:, 1].min() < -margin_y
        or quad[:, 1].max() > h + margin_y
    ):
        return None

    return {
        "quad": quad,
        "method": "canonical_blue_frame",
        "prior": 0.20,
        "debug_mask": mask,
    }


def _candidate_quads_from_binary(binary, image_shape):
    h, w = image_shape[:2]
    image_area = float(h * w)

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    proposals = []

    for contour in contours:
        area = cv2.contourArea(contour)
        area_ratio = area / image_area

        if not (0.06 <= area_ratio <= 0.98):
            continue

        perimeter = cv2.arcLength(contour, True)

        if perimeter <= 0:
            continue

        for eps in (0.012, 0.018, 0.025, 0.035, 0.05):
            approx = cv2.approxPolyDP(
                contour,
                eps * perimeter,
                True,
            )

            if len(approx) != 4:
                continue

            if not cv2.isContourConvex(approx):
                continue

            quad = order_points(
                approx.reshape(4, 2).astype(np.float32)
            )

            proposals.append(
                {
                    "quad": quad,
                    "method": "general_quad",
                    "prior": 0.0,
                }
            )

            break

    return proposals


def detect_general_quads(image):
    """
    Genera candidatos geométricos sin decidir inmediatamente
    cuál es el tablero.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0,
    )

    binaries = []

    # Canny.
    edges = cv2.Canny(
        blur,
        40,
        130,
    )

    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        np.ones((7, 7), np.uint8),
        iterations=2,
    )

    binaries.append(edges)

    # Adaptive threshold.
    adaptive = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        7,
    )

    adaptive = cv2.morphologyEx(
        adaptive,
        cv2.MORPH_CLOSE,
        np.ones((7, 7), np.uint8),
        iterations=2,
    )

    binaries.append(adaptive)

    # Otsu.
    _, otsu = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    binaries.append(otsu)

    proposals = []

    for binary in binaries:
        proposals.extend(
            _candidate_quads_from_binary(
                binary,
                image.shape,
            )
        )

    return proposals


def detect_bright_surface_quad(image):
    """
    Detecta una hoja/pantalla clara. Es un fallback,
    no el objetivo final.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    enhanced = clahe.apply(gray)

    candidates = []

    thresholds = []

    _, otsu = cv2.threshold(
        enhanced,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    thresholds.append(otsu)

    for percentile in (45, 55, 65, 72):
        value = float(
            np.percentile(
                enhanced,
                percentile,
            )
        )

        _, binary = cv2.threshold(
            enhanced,
            value,
            255,
            cv2.THRESH_BINARY,
        )

        thresholds.append(binary)

    h, w = gray.shape
    image_area = float(h * w)

    for binary in thresholds:
        work = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            np.ones((17, 17), np.uint8),
            iterations=2,
        )

        contours, _ = cv2.findContours(
            work,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        for contour in contours:
            area = cv2.contourArea(contour)
            area_ratio = area / image_area

            if not (0.10 <= area_ratio <= 0.98):
                continue

            hull = cv2.convexHull(contour)
            perimeter = cv2.arcLength(hull, True)

            for eps in (0.012, 0.02, 0.03, 0.05, 0.075):
                approx = cv2.approxPolyDP(
                    hull,
                    eps * perimeter,
                    True,
                )

                if len(approx) != 4:
                    continue

                if not cv2.isContourConvex(approx):
                    continue

                quad = order_points(
                    approx.reshape(4, 2).astype(np.float32)
                )

                candidates.append(
                    {
                        "quad": quad,
                        "method": "bright_surface",
                        "prior": -0.08,
                    }
                )

                break

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: quad_area(item["quad"]),
        reverse=True,
    )

    return candidates[0]


def _deduplicate_quads(candidates, image_shape):
    h, w = image_shape[:2]
    diagonal = math.hypot(w, h)

    unique = []

    for candidate in sorted(
        candidates,
        key=lambda item: (
            item.get("prior", 0.0),
            quad_area(item["quad"]),
        ),
        reverse=True,
    ):
        quad = order_points(candidate["quad"])

        duplicate = False

        for existing in unique:
            other = order_points(existing["quad"])

            mean_distance = float(
                np.linalg.norm(
                    quad - other,
                    axis=1,
                ).mean()
            )

            if mean_distance < 0.025 * diagonal:
                duplicate = True
                break

        if not duplicate:
            item = dict(candidate)
            item["quad"] = quad
            unique.append(item)

    return unique


def _deduplicate_rectangles(rectangles, center_distance=14.0):
    unique = []

    for item in sorted(
        rectangles,
        key=lambda value: value["area"],
        reverse=True,
    ):
        cx, cy = item["center"]

        duplicate = any(
            (cx - previous["center"][0]) ** 2
            + (cy - previous["center"][1]) ** 2
            <= center_distance ** 2
            for previous in unique
        )

        if not duplicate:
            unique.append(item)

    return unique


def _cluster_1d(values, k):
    data = np.asarray(
        values,
        dtype=np.float32,
    ).reshape(-1, 1)

    if len(data) < k:
        raise ValueError(
            "No hay suficientes puntos para agrupar."
        )

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        0.15,
    )

    compactness, labels, centers = cv2.kmeans(
        data,
        k,
        None,
        criteria,
        20,
        cv2.KMEANS_PP_CENTERS,
    )

    centers = centers.flatten()

    order = np.argsort(centers)

    remap = {
        int(old): int(new)
        for new, old in enumerate(order)
    }

    labels = np.array(
        [
            remap[int(label)]
            for label in labels.flatten()
        ],
        dtype=np.int32,
    )

    return (
        labels,
        centers[order],
        float(compactness),
    )


def detect_cell_grid_score(rectified, n):
    """
    No reconoce números.

    Solo responde:
    "¿este warp realmente se parece a una grilla
    Futoshiki de N×N?"
    """

    if n not in SUPPORTED_N:
        raise ValueError(
            f"n debe estar en {SUPPORTED_N}"
        )

    gray = cv2.cvtColor(
        rectified,
        cv2.COLOR_BGR2GRAY,
    )

    blur = cv2.GaussianBlur(
        gray,
        (3, 3),
        0,
    )

    binary = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        5,
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), np.uint8),
        iterations=1,
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    h, w = gray.shape
    image_area = float(h * w)

    rectangles = []

    for contour in contours:
        area = cv2.contourArea(contour)
        area_ratio = area / image_area

        if not (0.0015 <= area_ratio <= 0.12):
            continue

        perimeter = cv2.arcLength(
            contour,
            True,
        )

        if perimeter <= 0:
            continue

        approx = cv2.approxPolyDP(
            contour,
            0.03 * perimeter,
            True,
        )

        if len(approx) != 4:
            continue

        if not cv2.isContourConvex(approx):
            continue

        x, y, bw, bh = cv2.boundingRect(approx)

        if min(bw, bh) < 25:
            continue

        aspect = max(bw, bh) / max(min(bw, bh), 1e-6)

        if aspect > 1.65:
            continue

        rectangles.append(
            {
                "box": (x, y, x + bw, y + bh),
                "center": (
                    x + bw / 2.0,
                    y + bh / 2.0,
                ),
                "width": float(bw),
                "height": float(bh),
                "area": float(area),
            }
        )

    rectangles = _deduplicate_rectangles(
        rectangles,
        center_distance=max(10, int(900 / (n * 14))),
    )

    if len(rectangles) < n * 2:
        return {
            "score": 0.0,
            "coverage": 0.0,
            "reason": "too_few_cell_candidates",
            "candidate_count": len(rectangles),
        }

    widths = np.array(
        [item["width"] for item in rectangles],
        dtype=np.float64,
    )

    heights = np.array(
        [item["height"] for item in rectangles],
        dtype=np.float64,
    )

    med_w = float(np.median(widths))
    med_h = float(np.median(heights))

    rectangles = [
        item
        for item in rectangles
        if (
            0.45 * med_w
            <= item["width"]
            <= 1.85 * med_w
            and
            0.45 * med_h
            <= item["height"]
            <= 1.85 * med_h
        )
    ]

    if len(rectangles) < n * 2:
        return {
            "score": 0.0,
            "coverage": 0.0,
            "reason": "scale_filter_removed_too_many",
            "candidate_count": len(rectangles),
        }

    try:
        x_labels, x_centers, _ = _cluster_1d(
            [item["center"][0] for item in rectangles],
            n,
        )

        y_labels, y_centers, _ = _cluster_1d(
            [item["center"][1] for item in rectangles],
            n,
        )

    except Exception as error:
        return {
            "score": 0.0,
            "coverage": 0.0,
            "reason": str(error),
            "candidate_count": len(rectangles),
        }

    positions = {}

    for index, item in enumerate(rectangles):
        r = int(y_labels[index])
        c = int(x_labels[index])

        target = np.array(
            [
                x_centers[c],
                y_centers[r],
            ],
            dtype=np.float64,
        )

        current = np.array(
            item["center"],
            dtype=np.float64,
        )

        distance = float(
            np.linalg.norm(
                current - target
            )
        )

        key = (r, c)

        if (
            key not in positions
            or distance < positions[key]["distance"]
        ):
            positions[key] = {
                "distance": distance,
                "item": item,
            }

    coverage = len(positions) / float(n * n)

    dx = np.diff(x_centers)
    dy = np.diff(y_centers)

    spacing = np.concatenate(
        [
            dx if len(dx) else np.array([], dtype=np.float32),
            dy if len(dy) else np.array([], dtype=np.float32),
        ]
    )

    if len(spacing):
        spacing_cv = float(
            spacing.std()
            / max(
                spacing.mean(),
                1e-6,
            )
        )
    else:
        spacing_cv = 1.0

    # Ocupación del tablero. Si se detectó la hoja completa
    # en lugar del marco del puzzle, las celdas quedan demasiado
    # concentradas en el centro.
    x_span = float(
        x_centers[-1] - x_centers[0]
    ) if len(x_centers) > 1 else 0.0

    y_span = float(
        y_centers[-1] - y_centers[0]
    ) if len(y_centers) > 1 else 0.0

    span_ratio = min(
        x_span / max(w, 1),
        y_span / max(h, 1),
    )

    # En nuestros formatos canónicos, los centros extremos
    # suelen ocupar aproximadamente 65–80 % del warp.
    span_score = float(
        np.clip(
            (span_ratio - 0.42) / 0.25,
            0.0,
            1.0,
        )
    )

    regularity_score = float(
        np.clip(
            1.0 - spacing_cv / 0.35,
            0.0,
            1.0,
        )
    )

    accepted_widths = np.array(
        [
            value["item"]["width"]
            for value in positions.values()
        ],
        dtype=np.float64,
    )

    accepted_heights = np.array(
        [
            value["item"]["height"]
            for value in positions.values()
        ],
        dtype=np.float64,
    )

    if len(accepted_widths):
        size_cv = 0.5 * (
            accepted_widths.std()
            / max(accepted_widths.mean(), 1e-6)
            +
            accepted_heights.std()
            / max(accepted_heights.mean(), 1e-6)
        )
    else:
        size_cv = 1.0

    size_score = float(
        np.clip(
            1.0 - size_cv / 0.40,
            0.0,
            1.0,
        )
    )

    score = (
        0.55 * coverage
        + 0.20 * regularity_score
        + 0.15 * span_score
        + 0.10 * size_score
    )

    return {
        "score": float(score),
        "coverage": float(coverage),
        "candidate_count": len(rectangles),
        "detected_cells": len(positions),
        "spacing_cv": float(spacing_cv),
        "span_ratio": float(span_ratio),
        "regularity_score": regularity_score,
        "span_score": span_score,
        "size_score": size_score,
        "x_centers": x_centers,
        "y_centers": y_centers,
        "positions": positions,
    }


def _candidate_from_page_then_blue(image, output_size=1100):
    """
    Doble rectificación:
    foto -> hoja/pantalla -> marco azul.
    """

    page = detect_bright_surface_quad(image)

    if page is None:
        return None

    page_rect = warp_quad(
        image,
        page["quad"],
        output_size=output_size,
    )

    blue = detect_blue_frame_hough(
        page_rect["rectified"]
    )

    if blue is None:
        return None

    # Convertir el quad detectado en la hoja rectificada
    # de regreso a coordenadas de la foto original.
    quad_page = order_points(
        blue["quad"]
    ).reshape(-1, 1, 2)

    original_quad = cv2.perspectiveTransform(
        quad_page,
        page_rect["inverse_matrix"],
    ).reshape(4, 2)

    return {
        "quad": order_points(original_quad),
        "method": "page_then_blue_frame",
        "prior": 0.22,
    }


def detect_and_rectify_board(
    image,
    expected_n: Optional[int] = None,
    output_size=900,
    min_accept_score=0.58,
):
    """
    Detector final V5.

    Importante:
    NO acepta el primer cuadrilátero que encuentra.

    Genera múltiples candidatos y elige el que,
    después de rectificarse, se parece más a una
    grilla 4×4 o 5×5.
    """

    if expected_n is not None and expected_n not in SUPPORTED_N:
        raise ValueError(
            f"Visión V5 soporta solamente {SUPPORTED_N}."
        )

    candidates = []

    blue = detect_blue_frame_hough(image)

    if blue is not None:
        candidates.append(blue)

    page_blue = _candidate_from_page_then_blue(image)

    if page_blue is not None:
        candidates.append(page_blue)

    candidates.extend(
        detect_general_quads(image)
    )

    page = detect_bright_surface_quad(image)

    if page is not None:
        candidates.append(page)

    candidates = _deduplicate_quads(
        candidates,
        image.shape,
    )

    # Evitar explosión de candidatos.
    candidates = candidates[:35]

    if not candidates:
        raise ValueError(
            "No se encontraron candidatos de tablero."
        )

    n_values = (
        [expected_n]
        if expected_n is not None
        else list(SUPPORTED_N)
    )

    evaluated = []

    for candidate in candidates:
        try:
            rect = warp_quad(
                image,
                candidate["quad"],
                output_size=output_size,
            )

        except Exception:
            continue

        for n in n_values:
            info = detect_cell_grid_score(
                rect["rectified"],
                n,
            )

            total_score = (
                info["score"]
                +
                float(
                    candidate.get(
                        "prior",
                        0.0,
                    )
                )
            )

            evaluated.append(
                {
                    "candidate": candidate,
                    "n": int(n),
                    "grid_info": info,
                    "total_score": float(total_score),
                    "rectification": rect,
                }
            )

    if not evaluated:
        raise ValueError(
            "No fue posible evaluar los candidatos."
        )

    evaluated.sort(
        key=lambda item: item["total_score"],
        reverse=True,
    )

    best = evaluated[0]

    if best["grid_info"]["score"] < min_accept_score:
        raise ValueError(
            "Se localizaron regiones candidatas, pero ninguna "
            "se parece con suficiente confianza a una grilla "
            f"4×4/5×5. Mejor score={best['grid_info']['score']:.3f}"
        )

    result = dict(
        best["rectification"]
    )

    result.update(
        {
            "detector_method":
                best["candidate"]["method"],
            "detected_n":
                best["n"],
            "grid_detection_score":
                best["grid_info"]["score"],
            "grid_info":
                best["grid_info"],
            "total_detection_score":
                best["total_score"],
            "candidate_ranking":
                [
                    {
                        "method":
                            item["candidate"]["method"],
                        "n":
                            item["n"],
                        "grid_score":
                            item["grid_info"]["score"],
                        "total_score":
                            item["total_score"],
                        "coverage":
                            item["grid_info"].get(
                                "coverage",
                                0.0,
                            ),
                    }
                    for item in evaluated[:10]
                ],
        }
    )

    return result


def draw_board_detection(image, result):
    out = image.copy()

    contour = result[
        "board_contour"
    ]

    cv2.drawContours(
        out,
        [contour],
        -1,
        (0, 0, 255),
        max(
            3,
            image.shape[1] // 300,
        ),
    )

    quad = contour.reshape(4, 2)

    labels = [
        "TL",
        "TR",
        "BR",
        "BL",
    ]

    for point, label in zip(
        quad,
        labels,
    ):
        x, y = map(int, point)

        cv2.circle(
            out,
            (x, y),
            8,
            (0, 255, 255),
            -1,
        )

        cv2.putText(
            out,
            label,
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return out
