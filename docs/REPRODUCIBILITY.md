# Reproduction scope and known limitations

This record compares the supplied local source with Section 4.1 of the CGGL paper (DOI: 10.1016/j.inffus.2025.103292). The cleanup preserves algorithm settings and does not claim to reproduce every experiment.

## Paper settings versus supplied defaults

| Setting | Paper Section 4.1 | Supplied code |
| --- | --- | --- |
| CIFAR-100 clients / malicious clients | 50 / 10 | Inversion: 40 / 10; poisoning: 50 / 10 |
| Reconstructed batch size | 8 in main comparison | 8, with several fixed loops and shapes |
| Initial reconstruction rounds | 4 | `--n_seed=2` |
| Reconstruction iterations | 24,000 | `Main-Inversion.py`: 12,000 per reconstruction call |
| TV weight | 0.01 | `--tv=0.15` |
| BN weight | 0.01 | `--BN=0.001` |
| Group weight | 0.01 | `--Group=0.001` |
| Gradient noise standard deviation | 0.1 | `Main-Inversion.py`: 0.01 |
| Gradient sparsification | 90% set to zero | Compression branch passes `percentage=20`; its semantics need validation |
| Dataset split | 80% / 20% | CIFAR wrapper uses torchvision's standard train/test partitions |
| Datasets | CIFAR-100, ImageNet, LDCT | Active top-level scripts configure CIFAR-100; other branches are incomplete |

These are differences to resolve before reporting paper-matching results. They were not silently changed during repository cleanup.

## Resources and execution

- The original normal-model training block in `Main-CraftMaliciousModel.py` is commented out. Stage 1 loads `model/Cifar100_Normal_Model.pth` rather than generating it automatically.
- The custom diffusion checkpoint is supplied locally as `loss0.026.pth`; its training pipeline and a public hosting URL have not been established from this snapshot.
- Checkpoint sizes and hashes establish file identity, not numerical validity.
- The CUDA-specific calls remain. CPU or Apple Silicon execution is not validated.
- `guided_diffusion` imports MPI-related utilities. Finding the Python package does not establish that the system MPI library loads successfully.
- The ImageNet model heads, image sizes, diffusion checkpoint settings, and class labels must be checked together before running that branch. Replacing paths alone does not make the branch reproducible.
- The scripts use wildcard imports and execute at module scope. Use `run.py`; importing an experiment entry point can start training or downloads.
- Some command-line booleans in `inversefed/options.py` use legacy parsing rather than explicit boolean conversion. Passing the string `False` may not disable a feature.

## Evaluation details to verify

The supplied inversion script takes the minimum LPIPS value across the reconstructed batch and computes SSIM on a grid image. Those conventions need comparison with the intended paper evaluation before reporting aggregate metrics. Output filenames and some save loops also assume eight images.

The optional label-inference branch references `dgi_label_recon` and `compute_label_acc`, whose definitions were not found in the supplied source. The hook helper contains the original spelling `register_forwad_hook`. These branches were retained rather than replaced with guessed implementations.

Static inspection also found pre-existing undefined names in optional dataset/model/helper/example branches, including `image_tensor` in `utils.tensor2image_PIL`, `value`/`name` in `inversefed/data/loss.py`, and missing model definitions in `inversefed/nn/models.py`. No claim of a clean full linter or successful end-to-end run is made.

## Validation performed during cleanup

- Python source syntax parsing and bytecode compilation.
- Black formatting for top-level project code and new utilities.
- Launcher help and prerequisite-check execution without training.
- Local path configuration, including relative and absolute overrides.
- Comparison of original and cleaned Python ASTs to limit executable changes to path handling and directory preparation.
- A source-only archive and a check for accidentally included checkpoints, caches, logs, or IDE files.

No dataset download, GPU training, attack execution, full dependency installation, or comparison with paper metrics was performed. The target CUDA environment was unavailable on the cleanup host.
