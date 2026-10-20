"""Check dependencies and local resources without starting an experiment."""

import argparse
import importlib.metadata
import importlib.util
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cggl_paths import ROOT, DATA_DIR, CIFAR_DIFFUSION_CHECKPOINT

DISTRIBUTIONS = {
    "torch": "torch",
    "torchvision": "torchvision",
    "numpy": "numpy",
    "PIL": "Pillow",
    "matplotlib": "matplotlib",
    "cv2": "opencv-python",
    "skimage": "scikit-image",
    "scipy": "scipy",
    "lpips": "lpips",
    "blobfile": "blobfile",
    "mpi4py": "mpi4py",
    "tqdm": "tqdm",
    "six": "six",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=["craft", "poison", "invert"], default="invert"
    )
    parser.add_argument(
        "--runtime", action="store_true", help="Also import PyTorch and check CUDA."
    )
    args = parser.parse_args()
    failures = 0
    print(f"Repository: {ROOT}\nPython: {sys.version.split()[0]}")
    for module, distribution in DISTRIBUTIONS.items():
        installed = importlib.util.find_spec(module) is not None
        try:
            version = (
                importlib.metadata.version(distribution) if installed else "missing"
            )
        except importlib.metadata.PackageNotFoundError:
            version = "version unknown"
        print(
            f"{'OK' if installed else 'MISSING'} dependency {distribution}: {version}"
        )
        failures += not installed
    checkpoints = [ROOT / "model" / "Cifar100_Normal_Model.pth"]
    if args.stage in {"poison", "invert"}:
        checkpoints.append(ROOT / "model" / "Cifar100_FK_0_Malicious_Model.pth")
    if args.stage == "invert":
        checkpoints += [
            ROOT / "model" / "Cifar100_FK_0_MKrum_Poisoning_Model.pth",
            CIFAR_DIFFUSION_CHECKPOINT,
        ]
    for path in checkpoints:
        present = path.is_file() and path.stat().st_size > 0
        print(f"{'OK' if present else 'MISSING'} checkpoint {path}")
        failures += not present
    print(f"Dataset root: {DATA_DIR} (dataset completeness is not checked)")
    if args.runtime:
        try:
            import torch

            available = torch.cuda.is_available()
            print(f"CUDA available: {available}")
            failures += not available
        except Exception as error:
            print(f"PyTorch runtime error: {error}")
            failures += 1
    print("This checks prerequisites only; it does not validate numerical results.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
