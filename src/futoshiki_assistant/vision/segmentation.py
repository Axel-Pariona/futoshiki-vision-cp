import cv2
import numpy as np

from .models import normalize_binary_foreground


def segment_fixed(
    rectified,
    n,
    cell_inner_ratio=0.58,
    horizontal_width_ratio=0.24,
    horizontal_height_ratio=0.52,
    vertical_width_ratio=0.52,
    vertical_height_ratio=0.24,
):
    height, width = rectified.shape[:2]
    step = width / n

    def box(center_x, center_y, box_width, box_height):
        return (
            max(0, int(center_x - box_width / 2)),
            max(0, int(center_y - box_height / 2)),
            min(width, int(center_x + box_width / 2)),
            min(height, int(center_y + box_height / 2)),
        )

    cells = [
        [
            box(
                (col + 0.5) * step,
                (row + 0.5) * step,
                step * cell_inner_ratio,
                step * cell_inner_ratio,
            )
            for col in range(n)
        ]
        for row in range(n)
    ]

    horizontal = [
        [
            box(
                (col + 1) * step,
                (row + 0.5) * step,
                step * horizontal_width_ratio,
                step * horizontal_height_ratio,
            )
            for col in range(n - 1)
        ]
        for row in range(n)
    ]

    vertical = [
        [
            box(
                (col + 0.5) * step,
                (row + 1) * step,
                step * vertical_width_ratio,
                step * vertical_height_ratio,
            )
            for col in range(n)
        ]
        for row in range(n - 1)
    ]

    return {
        "raw_cell_boxes": cells,
        "cell_boxes": cells,
        "horizontal_boxes": horizontal,
        "vertical_boxes": vertical,
    }


def remove_grid_lines(rectified, n, line_ratio=0.55):
    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    binary = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2,
    )

    height, width = binary.shape
    cell_width = width / n
    cell_height = height / n

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(20, int(cell_width * line_ratio)), 1),
    )

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, max(20, int(cell_height * line_ratio))),
    )

    horizontal_lines = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        horizontal_kernel,
    )

    vertical_lines = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        vertical_kernel,
    )

    mask = cv2.bitwise_or(
        horizontal_lines,
        vertical_lines,
    )

    mask = cv2.dilate(
        mask,
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    )

    return cv2.bitwise_and(
        binary,
        cv2.bitwise_not(mask),
    )


def _deduplicate_rectangles(candidates, center_distance=14):
    unique = []

    for item in sorted(
        candidates,
        key=lambda value: value["area"],
        reverse=True,
    ):
        center_x, center_y = item["center"]

        duplicated = any(
            (
                (center_x - previous["center"][0]) ** 2
                + (center_y - previous["center"][1]) ** 2
                <= center_distance ** 2
            )
            for previous in unique
        )

        if not duplicated:
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
        cv2.TERM_CRITERIA_EPS
        + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        0.2,
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
        ]
    )

    return labels, centers[order], float(compactness)


def detect_cell_grid_adaptive(rectified, n):
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
        17,
        3,
    )

    closed = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    )

    contours, _ = cv2.findContours(
        closed,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    height, width = gray.shape
    image_area = float(height * width)
    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)
        area_ratio = area / image_area

        if not (0.0015 <= area_ratio <= 0.12):
            continue

        perimeter = cv2.arcLength(contour, True)

        approx = cv2.approxPolyDP(
            contour,
            0.03 * perimeter,
            True,
        )

        if len(approx) != 4:
            continue

        if not cv2.isContourConvex(approx):
            continue

        x, y, box_width, box_height = cv2.boundingRect(
            approx
        )

        if min(box_width, box_height) < 20:
            continue

        aspect = max(
            box_width,
            box_height,
        ) / max(
            min(box_width, box_height),
            1e-6,
        )

        if aspect > 1.75:
            continue

        candidates.append(
            {
                "box": (
                    x,
                    y,
                    x + box_width,
                    y + box_height,
                ),
                "center": (
                    x + box_width / 2,
                    y + box_height / 2,
                ),
                "width": float(box_width),
                "height": float(box_height),
                "area": float(area),
            }
        )

    candidates = _deduplicate_rectangles(candidates)
    minimum = max(n * n // 2, n * 2)

    if len(candidates) < minimum:
        return {
            "success": False,
            "reason": "Pocos rectángulos candidatos.",
            "candidate_count": len(candidates),
        }

    median_width = float(
        np.median(
            [item["width"] for item in candidates]
        )
    )

    median_height = float(
        np.median(
            [item["height"] for item in candidates]
        )
    )

    filtered = [
        item
        for item in candidates
        if (
            0.45 * median_width
            <= item["width"]
            <= 1.9 * median_width
            and
            0.45 * median_height
            <= item["height"]
            <= 1.9 * median_height
        )
    ]

    if len(filtered) < minimum:
        filtered = candidates

    try:
        x_labels, x_centers, _ = _cluster_1d(
            [item["center"][0] for item in filtered],
            n,
        )

        y_labels, y_centers, _ = _cluster_1d(
            [item["center"][1] for item in filtered],
            n,
        )
    except ValueError as error:
        return {
            "success": False,
            "reason": str(error),
            "candidate_count": len(filtered),
        }

    buckets = {
        (row, col): []
        for row in range(n)
        for col in range(n)
    }

    for index, item in enumerate(filtered):
        row = int(y_labels[index])
        col = int(x_labels[index])

        target = np.array(
            [x_centers[col], y_centers[row]],
            dtype=float,
        )

        current = np.array(
            item["center"],
            dtype=float,
        )

        candidate = dict(item)
        candidate["distance"] = float(
            np.linalg.norm(target - current)
        )

        buckets[(row, col)].append(candidate)

    detected = {}

    for key, items in buckets.items():
        if items:
            detected[key] = min(
                items,
                key=lambda item: item["distance"],
            )["box"]

    coverage = len(detected) / float(n * n)

    if detected:
        cell_width = float(
            np.median(
                [
                    box[2] - box[0]
                    for box in detected.values()
                ]
            )
        )
        cell_height = float(
            np.median(
                [
                    box[3] - box[1]
                    for box in detected.values()
                ]
            )
        )
    else:
        cell_width = median_width
        cell_height = median_height

    boxes = []
    inferred_count = 0

    for row in range(n):
        row_boxes = []

        for col in range(n):
            if (row, col) in detected:
                current_box = detected[(row, col)]
            else:
                inferred_count += 1

                center_x = float(x_centers[col])
                center_y = float(y_centers[row])

                x1 = int(round(center_x - cell_width / 2))
                y1 = int(round(center_y - cell_height / 2))
                x2 = int(round(center_x + cell_width / 2))
                y2 = int(round(center_y + cell_height / 2))

                current_box = (
                    max(0, x1),
                    max(0, y1),
                    min(width, x2),
                    min(height, y2),
                )

            row_boxes.append(current_box)

        boxes.append(row_boxes)

    spacing = np.concatenate(
        [
            np.diff(x_centers),
            np.diff(y_centers),
        ]
    )

    spacing_cv = (
        float(
            spacing.std()
            / max(spacing.mean(), 1e-6)
        )
        if len(spacing)
        else 0.0
    )

    success = (
        coverage >= 0.55
        and spacing_cv <= 0.35
    )

    return {
        "success": bool(success),
        "boxes": boxes,
        "x_centers": x_centers,
        "y_centers": y_centers,
        "coverage": float(coverage),
        "detected_count": len(detected),
        "inferred_count": inferred_count,
        "candidate_count": len(filtered),
        "spacing_cv": spacing_cv,
        "cell_width": cell_width,
        "cell_height": cell_height,
    }


def _shrink_box(box, ratio=0.17):
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    dx = int(round(width * ratio))
    dy = int(round(height * ratio))

    return (
        x1 + dx,
        y1 + dy,
        x2 - dx,
        y2 - dy,
    )


def build_adaptive_roi_boxes(
    grid_info,
    image_shape,
    n,
):
    height, width = image_shape[:2]
    raw = grid_info["boxes"]

    cells = [
        [
            _shrink_box(raw[row][col])
            for col in range(n)
        ]
        for row in range(n)
    ]

    horizontal = []

    for row in range(n):
        current_row = []

        for col in range(n - 1):
            left = raw[row][col]
            right = raw[row][col + 1]

            left_center = (
                (left[0] + left[2]) / 2,
                (left[1] + left[3]) / 2,
            )

            right_center = (
                (right[0] + right[2]) / 2,
                (right[1] + right[3]) / 2,
            )

            center_x = int(
                round(
                    (left_center[0] + right_center[0]) / 2
                )
            )

            center_y = int(
                round(
                    (left_center[1] + right_center[1]) / 2
                )
            )

            center_distance = abs(
                right_center[0] - left_center[0]
            )

            mean_height = (
                (left[3] - left[1])
                + (right[3] - right[1])
            ) / 2

            roi_width = max(
                12,
                int(round(center_distance * 0.28)),
            )

            roi_height = max(
                12,
                int(round(mean_height * 0.52)),
            )

            current_row.append(
                (
                    max(0, center_x - roi_width // 2),
                    max(0, center_y - roi_height // 2),
                    min(width, center_x + roi_width // 2),
                    min(height, center_y + roi_height // 2),
                )
            )

        horizontal.append(current_row)

    vertical = []

    for row in range(n - 1):
        current_row = []

        for col in range(n):
            top = raw[row][col]
            bottom = raw[row + 1][col]

            top_center = (
                (top[0] + top[2]) / 2,
                (top[1] + top[3]) / 2,
            )

            bottom_center = (
                (bottom[0] + bottom[2]) / 2,
                (bottom[1] + bottom[3]) / 2,
            )

            center_x = int(
                round(
                    (top_center[0] + bottom_center[0]) / 2
                )
            )

            center_y = int(
                round(
                    (top_center[1] + bottom_center[1]) / 2
                )
            )

            center_distance = abs(
                bottom_center[1] - top_center[1]
            )

            mean_width = (
                (top[2] - top[0])
                + (bottom[2] - bottom[0])
            ) / 2

            roi_width = max(
                12,
                int(round(mean_width * 0.52)),
            )

            roi_height = max(
                12,
                int(round(center_distance * 0.28)),
            )

            current_row.append(
                (
                    max(0, center_x - roi_width // 2),
                    max(0, center_y - roi_height // 2),
                    min(width, center_x + roi_width // 2),
                    min(height, center_y + roi_height // 2),
                )
            )

        vertical.append(current_row)

    return {
        "raw_cell_boxes": raw,
        "cell_boxes": cells,
        "horizontal_boxes": horizontal,
        "vertical_boxes": vertical,
    }


def _crop_box(image, box):
    x1, y1, x2, y2 = box
    return image[y1:y2, x1:x2]


def extract_rois(rectified, roi_boxes, n):
    content = remove_grid_lines(rectified, n)

    cells = [
        [
            _crop_box(content, box)
            for box in row
        ]
        for row in roi_boxes["cell_boxes"]
    ]

    horizontal = [
        [
            _crop_box(content, box)
            for box in row
        ]
        for row in roi_boxes["horizontal_boxes"]
    ]

    vertical = [
        [
            _crop_box(content, box)
            for box in row
        ]
        for row in roi_boxes["vertical_boxes"]
    ]

    return {
        "content": content,
        "cell_rois": cells,
        "horizontal_rois": horizontal,
        "vertical_rois": vertical,
    }


def segment_rectified_board(
    rectified,
    n,
    mode="auto",
):
    if mode not in ("auto", "adaptive", "fixed"):
        raise ValueError(
            "mode debe ser auto, adaptive o fixed."
        )

    grid_info = None

    if mode in ("auto", "adaptive"):
        grid_info = detect_cell_grid_adaptive(
            rectified,
            n,
        )

        if grid_info.get("success", False):
            roi_boxes = build_adaptive_roi_boxes(
                grid_info,
                rectified.shape,
                n,
            )

            return {
                "method": "adaptive",
                "grid_info": grid_info,
                "roi_boxes": roi_boxes,
                "rois": extract_rois(
                    rectified,
                    roi_boxes,
                    n,
                ),
            }

        if mode == "adaptive":
            raise ValueError(
                "La segmentación adaptable no alcanzó confianza suficiente."
            )

    roi_boxes = segment_fixed(
        rectified,
        n,
    )

    return {
        "method": "fixed",
        "grid_info": grid_info,
        "roi_boxes": roi_boxes,
        "rois": extract_rois(
            rectified,
            roi_boxes,
            n,
        ),
    }
