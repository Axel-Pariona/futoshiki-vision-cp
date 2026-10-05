from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn


DIGIT_CLASSES = ["1", "2", "3", "4", "5", "6"]
INEQUALITY_CLASSES = ["blank", "<", ">"]
ROI_SIZE = 64


class SmallFutoshikiCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 96),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(96, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


@dataclass
class ModelBundle:
    digit_model: nn.Module
    inequality_model: nn.Module
    metadata: dict
    device: torch.device


def load_model_bundle(models_dir, device=None):
    models_dir = Path(models_dir)

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    metadata_path = models_dir / "futoshiki_vision_v2_config.json"
    digit_path = models_dir / "futoshiki_digit_model_v2.pt"
    inequality_path = models_dir / "futoshiki_inequality_model_v2.pt"

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    digit_checkpoint = torch.load(
        digit_path,
        map_location=device,
    )

    inequality_checkpoint = torch.load(
        inequality_path,
        map_location=device,
    )

    digit_model = SmallFutoshikiCNN(
        len(digit_checkpoint["classes"])
    ).to(device)

    inequality_model = SmallFutoshikiCNN(
        len(inequality_checkpoint["classes"])
    ).to(device)

    digit_model.load_state_dict(
        digit_checkpoint["state_dict"]
    )

    inequality_model.load_state_dict(
        inequality_checkpoint["state_dict"]
    )

    digit_model.eval()
    inequality_model.eval()

    return ModelBundle(
        digit_model=digit_model,
        inequality_model=inequality_model,
        metadata=metadata,
        device=device,
    )


def normalize_binary_foreground(roi):
    if roi is None or roi.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    if roi.ndim == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi.copy()

    if len(np.unique(gray)) > 8:
        _, binary = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
    else:
        binary = gray.copy()

    if binary.mean() > 127:
        binary = cv2.bitwise_not(binary)

    return binary


def roi_presence_score(roi, border_ratio=0.08):
    binary = normalize_binary_foreground(roi)
    height, width = binary.shape

    if height <= 2 or width <= 2:
        return 0.0

    border_x = max(1, int(width * border_ratio))
    border_y = max(1, int(height * border_ratio))

    work = binary.copy()
    work[:border_y, :] = 0
    work[height - border_y :, :] = 0
    work[:, :border_x] = 0
    work[:, width - border_x :] = 0

    work = cv2.morphologyEx(
        work,
        cv2.MORPH_OPEN,
        np.ones((2, 2), dtype=np.uint8),
    )

    count, _, stats, _ = cv2.connectedComponentsWithStats(
        (work > 0).astype(np.uint8),
        8,
    )

    if count <= 1:
        return 0.0

    largest = stats[1:, cv2.CC_STAT_AREA].max()
    return float(largest) / float(height * width)


def prepare_roi_for_model(
    roi,
    output_size=ROI_SIZE,
    padding=5,
):
    binary = normalize_binary_foreground(roi)
    coordinates = cv2.findNonZero(binary)

    if coordinates is None:
        return np.zeros(
            (output_size, output_size),
            dtype=np.uint8,
        )

    x, y, width, height = cv2.boundingRect(coordinates)
    crop = binary[y : y + height, x : x + width]

    usable = output_size - 2 * padding
    scale = min(
        usable / max(width, 1),
        usable / max(height, 1),
    )

    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    resized = cv2.resize(
        crop,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.zeros(
        (output_size, output_size),
        dtype=np.uint8,
    )

    x0 = (output_size - new_width) // 2
    y0 = (output_size - new_height) // 2

    canvas[
        y0 : y0 + new_height,
        x0 : x0 + new_width,
    ] = resized

    return canvas


def predict_prepared_image(
    model,
    prepared,
    class_names,
    device,
):
    tensor = torch.from_numpy(
        prepared.astype(np.float32) / 255.0
    )

    tensor = (
        tensor.unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        probabilities = torch.softmax(
            model(tensor),
            dim=1,
        )[0]

    index = int(probabilities.argmax().item())

    return {
        "label": class_names[index],
        "confidence": float(
            probabilities[index].item()
        ),
        "probabilities": {
            class_names[i]: float(
                probabilities[i].item()
            )
            for i in range(len(class_names))
        },
    }
