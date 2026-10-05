import numpy as np

from futoshiki_assistant.vision.models import roi_presence_score


def test_blank_presence_is_zero():
    roi = np.zeros(
        (64, 64),
        dtype=np.uint8,
    )

    assert roi_presence_score(roi) == 0.0


def test_foreground_presence_is_positive():
    roi = np.zeros(
        (64, 64),
        dtype=np.uint8,
    )

    roi[20:44, 28:36] = 255

    assert roi_presence_score(roi) > 0.0
