import platform

import cv2
import matplotlib
import numpy
import pandas
import torch

try:
    import ortools
except ImportError:
    ortools = None


def main():
    versions = {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "opencv": cv2.__version__,
        "torch": torch.__version__,
        "pandas": pandas.__version__,
        "matplotlib": matplotlib.__version__,
        "ortools": getattr(ortools, "__version__", "not-installed"),
    }

    for name, version in versions.items():
        print(f"{name}: {version}")


if __name__ == "__main__":
    main()
