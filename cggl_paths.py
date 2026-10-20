"""Local resource paths. Importing this module does not create directories."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _path(variable, default):
    value = Path(os.environ.get(variable, str(default))).expanduser()
    return value if value.is_absolute() else ROOT / value


DATA_DIR = _path("CGGL_DATA_DIR", ROOT / "dataset")
IMAGES_DIR = _path("CGGL_IMAGES_DIR", ROOT / "images")
CIFAR_DIFFUSION_CHECKPOINT = _path(
    "CGGL_CIFAR_DIFFUSION_CHECKPOINT", ROOT / "checkpoints" / "loss0.026.pth"
)
IMAGENET_DIFFUSION_CHECKPOINT = _path(
    "CGGL_IMAGENET_DIFFUSION_CHECKPOINT", ROOT / "checkpoints" / "256x256_diffusion.pt"
)
IMAGENET_CLASSIFIER_CHECKPOINT = _path(
    "CGGL_IMAGENET_CLASSIFIER_CHECKPOINT",
    ROOT / "checkpoints" / "256x256_classifier.pt",
)


def prepare_directories():
    """Create writable folders required by the original experiment scripts."""
    for folder in (
        DATA_DIR,
        IMAGES_DIR,
        ROOT / "model",
        ROOT / "checkpoints",
        ROOT / "result",
        ROOT / "output",
        ROOT / "temp_data",
        ROOT / "tables",
        ROOT / "log" / "S1",
        ROOT / "log" / "S2",
        ROOT / "log" / "S3",
    ):
        folder.mkdir(parents=True, exist_ok=True)
