import cv2

from .models import (
    DIGIT_CLASSES,
    INEQUALITY_CLASSES,
    prepare_roi_for_model,
    predict_prepared_image,
    roi_presence_score,
)


def classify_digit(
    roi,
    n,
    model_bundle,
    presence_threshold,
):
    presence = float(
        roi_presence_score(roi)
    )

    if presence < presence_threshold:
        return {
            "label": "blank",
            "confidence": 1.0,
            "presence_score": presence,
            "accepted": False,
        }

    prediction = predict_prepared_image(
        model_bundle.digit_model,
        prepare_roi_for_model(roi),
        DIGIT_CLASSES,
        model_bundle.device,
    )

    value = int(prediction["label"])
    accepted = 1 <= value <= n

    return {
        **prediction,
        "presence_score": presence,
        "accepted": accepted,
    }


def classify_inequality(
    roi,
    model_bundle,
    min_confidence,
    margin_vs_blank,
    rotate_vertical=False,
):
    work = roi

    if rotate_vertical:
        work = cv2.rotate(
            work,
            cv2.ROTATE_90_COUNTERCLOCKWISE,
        )

    prediction = predict_prepared_image(
        model_bundle.inequality_model,
        prepare_roi_for_model(work),
        INEQUALITY_CLASSES,
        model_bundle.device,
    )

    probabilities = prediction["probabilities"]
    p_blank = float(probabilities["blank"])
    p_lt = float(probabilities["<"])
    p_gt = float(probabilities[">"])

    if p_lt >= p_gt:
        best_symbol = "<"
        symbol_confidence = p_lt
    else:
        best_symbol = ">"
        symbol_confidence = p_gt

    margin = symbol_confidence - p_blank

    accepted = (
        symbol_confidence >= min_confidence
        and margin >= margin_vs_blank
    )

    return {
        "label": best_symbol if accepted else "blank",
        "raw_label": prediction["label"],
        "confidence": float(prediction["confidence"]),
        "probabilities": probabilities,
        "blank_probability": p_blank,
        "symbol_confidence": symbol_confidence,
        "margin_vs_blank": margin,
        "accepted": accepted,
    }


def recognize_board(
    segmentation,
    n,
    model_bundle,
    digit_presence_threshold=0.010,
    inequality_min_confidence=0.75,
    inequality_margin_vs_blank=0.20,
):
    rois = segmentation["rois"]

    givens = []
    inequalities = []

    details = {
        "cells": [],
        "horizontal": [],
        "vertical": [],
    }

    for row, roi_row in enumerate(rois["cell_rois"]):
        detail_row = []

        for col, roi in enumerate(roi_row):
            prediction = classify_digit(
                roi,
                n,
                model_bundle,
                digit_presence_threshold,
            )

            detail_row.append(prediction)

            if prediction["accepted"]:
                givens.append(
                    {
                        "row": row,
                        "col": col,
                        "value": int(prediction["label"]),
                    }
                )

        details["cells"].append(detail_row)

    for row, roi_row in enumerate(
        rois["horizontal_rois"]
    ):
        detail_row = []

        for col, roi in enumerate(roi_row):
            prediction = classify_inequality(
                roi,
                model_bundle,
                inequality_min_confidence,
                inequality_margin_vs_blank,
                rotate_vertical=False,
            )

            detail_row.append(prediction)

            if prediction["accepted"]:
                inequalities.append(
                    {
                        "cell1": [row, col],
                        "operator": prediction["label"],
                        "cell2": [row, col + 1],
                    }
                )

        details["horizontal"].append(detail_row)

    for row, roi_row in enumerate(
        rois["vertical_rois"]
    ):
        detail_row = []

        for col, roi in enumerate(roi_row):
            prediction = classify_inequality(
                roi,
                model_bundle,
                inequality_min_confidence,
                inequality_margin_vs_blank,
                rotate_vertical=True,
            )

            detail_row.append(prediction)

            if prediction["accepted"]:
                inequalities.append(
                    {
                        "cell1": [row, col],
                        "operator": prediction["label"],
                        "cell2": [row + 1, col],
                    }
                )

        details["vertical"].append(detail_row)

    return {
        "instance": {
            "type": "futoshiki",
            "size": int(n),
            "givens": givens,
            "inequalities": inequalities,
        },
        "details": details,
    }
