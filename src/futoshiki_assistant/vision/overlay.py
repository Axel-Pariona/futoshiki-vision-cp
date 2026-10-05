import cv2


def _draw_box(image, box, color, thickness=2):
    x1, y1, x2, y2 = box
    cv2.rectangle(
        image,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        color,
        thickness,
    )


def draw_segmentation_overlay(
    rectified,
    roi_boxes,
    n,
):
    output = rectified.copy()

    for row in range(n):
        for col in range(n):
            raw = roi_boxes["raw_cell_boxes"][row][col]
            inner = roi_boxes["cell_boxes"][row][col]

            _draw_box(output, raw, (40, 180, 40), 2)
            _draw_box(output, inner, (180, 120, 20), 1)

            x1, y1, _, _ = raw

            cv2.putText(
                output,
                f"{row},{col}",
                (int(x1) + 4, int(y1) + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

    for row in roi_boxes["horizontal_boxes"]:
        for box in row:
            _draw_box(output, box, (255, 170, 0), 1)

    for row in roi_boxes["vertical_boxes"]:
        for box in row:
            _draw_box(output, box, (180, 0, 180), 1)

    return output


def draw_perception_overlay(
    rectified,
    roi_boxes,
    details,
    n,
):
    output = rectified.copy()

    for row in range(n):
        for col in range(n):
            box = roi_boxes["raw_cell_boxes"][row][col]
            item = details["cells"][row][col]
            label = item["label"]

            presence = float(
                item.get("presence_score", 0.0)
            )

            confidence = float(
                item.get("confidence", 0.0)
            )

            color = (
                (150, 150, 150)
                if label == "blank"
                else (40, 180, 40)
            )

            _draw_box(output, box, color, 2)

            x1, y1, _, y2 = box

            cv2.putText(
                output,
                f"({row},{col}) {label}",
                (int(x1) + 3, int(y1) + 17),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                1,
                cv2.LINE_AA,
            )

            cv2.putText(
                output,
                f"p={presence:.3f} c={confidence:.2f}",
                (int(x1) + 3, int(y2) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                color,
                1,
                cv2.LINE_AA,
            )

    for orientation, key in (
        ("horizontal", "horizontal_boxes"),
        ("vertical", "vertical_boxes"),
    ):
        for row, detail_row in enumerate(details[orientation]):
            for col, item in enumerate(detail_row):
                box = roi_boxes[key][row][col]
                label = item["label"]
                confidence = float(
                    item.get(
                        "symbol_confidence",
                        item.get("confidence", 0.0),
                    )
                )

                color = (
                    (170, 170, 170)
                    if label == "blank"
                    else (
                        (255, 120, 0)
                        if orientation == "horizontal"
                        else (180, 0, 180)
                    )
                )

                _draw_box(output, box, color, 1)

                x1, y1, x2, y2 = box
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                cv2.putText(
                    output,
                    f"{label} {confidence:.2f}",
                    (center_x - 18, center_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.40,
                    color,
                    1,
                    cv2.LINE_AA,
                )

    return output


def draw_solution_rectified(
    rectified,
    solution,
    givens,
    n,
):
    output = rectified.copy()
    fixed = {
        (item["row"], item["col"])
        for item in givens
    }

    cell_size = output.shape[0] / n

    for row in range(n):
        for col in range(n):
            if (row, col) in fixed:
                continue

            value = str(solution[row][col])
            center_x = int((col + 0.5) * cell_size)
            center_y = int((row + 0.5) * cell_size)

            scale = max(0.8, cell_size / 115)
            thickness = max(2, int(scale * 2))

            text_size, _ = cv2.getTextSize(
                value,
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                thickness,
            )

            cv2.putText(
                output,
                value,
                (
                    center_x - text_size[0] // 2,
                    center_y + text_size[1] // 2,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                (40, 80, 180),
                thickness,
                cv2.LINE_AA,
            )

    return output


def project_overlay_to_original(
    rect_info,
    overlay,
):
    rectified = rect_info["rectified"]

    delta = cv2.absdiff(
        overlay,
        rectified,
    )

    mask = cv2.cvtColor(
        delta,
        cv2.COLOR_BGR2GRAY,
    )

    _, mask = cv2.threshold(
        mask,
        5,
        255,
        cv2.THRESH_BINARY,
    )

    original = rect_info["original"]
    target_shape = (
        original.shape[1],
        original.shape[0],
    )

    warped_overlay = cv2.warpPerspective(
        overlay,
        rect_info["inverse_matrix"],
        target_shape,
    )

    warped_mask = cv2.warpPerspective(
        mask,
        rect_info["inverse_matrix"],
        target_shape,
    )

    output = original.copy()
    output[warped_mask > 0] = warped_overlay[
        warped_mask > 0
    ]

    return output
