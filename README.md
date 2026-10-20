# CGGL

**A client-side generative gradient leakage attack with double diffusion prior**  
Bin Pu, Zhizhi Liu, Liwen Wu, Kai Xu, Bocheng Liang, Ziyang He, Benteng Ma, Lei Zhao  
*Information Fusion*, 123 (2025), 103292.

[Paper](https://doi.org/10.1016/j.inffus.2025.103292) · [中文说明](README.zh-CN.md) · [Reproduction notes](docs/REPRODUCIBILITY.md)

CGGL combines client-side model poisoning and diffusion priors to study image reconstruction from aggregated gradients in federated learning.

This repository contains a cleaned research code snapshot. The supplied scripts primarily configure CIFAR-100; datasets, pretrained weights, and historical results are distributed separately from source code. Full paper reproduction has **not** been validated for this snapshot. Parameter differences and incomplete branches are documented in the [reproduction notes](docs/REPRODUCIBILITY.md).

## Repository layout

```text
CGGL/
├── Main-CraftMaliciousModel.py   # Stage 1: construct a malicious model
├── Main-PoisoningGlobalModel.py  # Stage 2: poison the global model
├── Main-Inversion.py             # Stage 3: reconstruct images
├── Main_framework_lib*.py        # Experiment-stage helpers
├── dataset.py                   # Dataset wrappers and sampling
├── federated_learning.py        # Federated learning and aggregation
├── model_structure.py           # Model definitions
├── defense.py                   # Gradient defenses
├── inversefed/                  # Modified gradient reconstruction library
├── guided_diffusion/            # Guided diffusion components
├── simpleAICV/                  # Diffusion models and supporting components
├── cggl_paths.py                # Configurable resource paths
├── run.py                       # Launcher with repository-root working directory
├── scripts/check_environment.py # Dependency and checkpoint checks
├── checkpoints/                 # Local diffusion checkpoints (ignored by Git)
├── model/                       # Local FL checkpoints (ignored by Git)
└── docs/                        # Reproduction and environment notes
```

## Environment

The supplied environment used Python 3.9, PyTorch 2.0.0 with CUDA 11.7, and torchvision 0.15.1. The paper reports a Linux server with an NVIDIA RTX 4090. The reconstruction code contains explicit CUDA operations; use a compatible NVIDIA GPU for experiments.

From the repository root:

```bash
conda env create -f environment.yml
conda activate cggl
```

The new environment file removes Windows-specific build identifiers and the local Conda prefix. Package versions in `requirements.txt` come from the supplied environment; this installation recipe has not been independently validated. The sanitized original export is retained as [`docs/environment.original-windows.yml`](docs/environment.original-windows.yml).

For an existing Python 3.9 environment, install a compatible PyTorch 2.0.0 / torchvision 0.15.1 pair and an MPI runtime, then run `python -m pip install -r requirements.txt`. Refer to the [official PyTorch version instructions](https://pytorch.org/get-started/previous-versions/) for CUDA wheels.

## Prepare local resources

1. Put `loss0.026.pth` in `checkpoints/`.
2. Put `Cifar100_Normal_Model.pth` in `model/` before Stage 1. Stages 2 and 3 require the malicious model; Stage 3 also requires the poisoned global model. Existing supplied checkpoints can be used to skip the preceding stages.
3. The CIFAR-100 wrapper downloads data into `dataset/` when an experiment starts. See [`dataset/README.md`](dataset/README.md) for the resulting layout.

See [`checkpoints/README.md`](checkpoints/README.md) for file hashes and the optional ImageNet branch. No public download URL for the supplied custom weights has been established.

Paths default to folders inside this repository. Override them before launching with these environment variables:

| Variable | Default |
| --- | --- |
| `CGGL_DATA_DIR` | `dataset/` |
| `CGGL_IMAGES_DIR` | `images/` |
| `CGGL_CIFAR_DIFFUSION_CHECKPOINT` | `checkpoints/loss0.026.pth` |
| `CGGL_IMAGENET_DIFFUSION_CHECKPOINT` | `checkpoints/256x256_diffusion.pt` |
| `CGGL_IMAGENET_CLASSIFIER_CHECKPOINT` | `checkpoints/256x256_classifier.pt` |

Relative overrides are resolved against the repository root. These variables are read when the process starts. Legacy FL checkpoint paths remain under `model/`.

## Run the supplied experiments

Check prerequisites first:

```bash
python scripts/check_environment.py --stage invert --runtime
```

A nonzero exit status means a required dependency, checkpoint, or CUDA runtime is unavailable. This check does not start training, verify dataset contents, or certify reproduction.

Run the stages in order, or start from existing stage checkpoints:

```bash
python run.py craft
python run.py poison
python run.py invert
```

`run.py` creates output folders and runs each original script from the repository root. The original `Main-*.py` filenames remain valid. The old `.ipynb` files mentioned in the supplied README were not present.

Experiment settings remain in the original scripts and `inversefed/options.py`. The checked-in inversion path assumes **8 images** in several places. Do not treat `--num_images` as a complete batch-size switch. See the reproduction notes before changing dataset, batch size, defense, or model settings.

## Source checks

These checks require no GPU or datasets:

```bash
python run.py --help
python -m compileall -q .
python -m pip install -r requirements-dev.txt
python -m black --check .
```

GitHub Actions runs formatting, Python 3.9 syntax parsing, and launcher help. These checks do not run the experiments or validate numerical results. Vendored libraries retain their upstream formatting.

## Citation

```bibtex
@article{pu2025cggl,
  title   = {CGGL: A client-side generative gradient leakage attack with double diffusion prior},
  author  = {Pu, Bin and Liu, Zhizhi and Wu, Liwen and Xu, Kai and Liang, Bocheng and He, Ziyang and Ma, Benteng and Zhao, Lei},
  journal = {Information Fusion},
  volume  = {123},
  pages   = {103292},
  year    = {2025},
  doi     = {10.1016/j.inffus.2025.103292}
}
```

## Acknowledgments and licensing

The supplied project includes modified or vendored components from [Inverting Gradients](https://github.com/JonasGeiping/invertinggradients), [Guided Diffusion](https://github.com/openai/guided-diffusion), and [SimpleAICV](https://github.com/zgcr/SimpleAICV_pytorch_training_examples). Its original folder name and README also refer to PGLC. See [third-party notices](THIRD_PARTY_NOTICES.md) for provenance and retained license texts.

A project-wide license for the original CGGL/PGLC-derived code was not present in the supplied files. This cleanup does not assign a new license to that code.
