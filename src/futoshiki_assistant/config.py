from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class VisionSettings:
    supported_sizes: Tuple[int, ...] = (4, 5)
    output_size: int = 900
    segmentation_mode: str = "auto"

    digit_presence_threshold: float = 0.010
    inequality_min_confidence: float = 0.75
    inequality_margin_vs_blank: float = 0.20

    detection_score_fail: float = 0.70
    detection_score_warning: float = 0.85
    detection_coverage_fail: float = 0.80

    scope_min_board_area_ratio: float = 0.18
    scope_max_board_area_ratio: float = 0.80
    scope_min_margin_ratio: float = 0.04
    scope_max_perspective_ratio: float = 1.20
    scope_min_blur_score: float = 120.0
    scope_min_brightness: float = 65.0

    def to_dict(self):
        data = asdict(self)
        data["supported_sizes"] = list(self.supported_sizes)
        return data

    @classmethod
    def from_json(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if "supported_sizes" in data:
            data["supported_sizes"] = tuple(data["supported_sizes"])
        return cls(**data)

    def save_json(self, path):
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
