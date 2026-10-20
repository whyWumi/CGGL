"""Launch a legacy experiment from the repository root."""

import argparse
import subprocess
import sys

from cggl_paths import ROOT, prepare_directories

SCRIPTS = {
    "craft": "Main-CraftMaliciousModel.py",
    "poison": "Main-PoisoningGlobalModel.py",
    "invert": "Main-Inversion.py",
}


def main():
    parser = argparse.ArgumentParser(
        description="Launch a CGGL experiment after preparing local output folders.",
        epilog="Requires the research environment, data, and checkpoints. See README.md.",
    )
    parser.add_argument("stage", choices=SCRIPTS, help="Experiment stage to execute.")
    args, remaining = parser.parse_known_args()
    prepare_directories()
    return subprocess.call(
        [sys.executable, str(ROOT / SCRIPTS[args.stage]), *remaining], cwd=ROOT
    )


if __name__ == "__main__":
    raise SystemExit(main())
