# Checkpoints

Model binaries are excluded from Git. No hosted URL for the custom checkpoints has been established. The file list below refers to the originally supplied local artifacts.

| Filename | Bytes | SHA-256 |
| --- | ---: | --- |
| `Cifar100_FK_0_MKrum_Poisoning_Model.pth` | 44,963,063 | `6ce999f3527f1e6226f3a22c0159968ee0e6bb8e239f62d37c41bb1c26172917` |
| `Cifar100_FK_0_Malicious_Model.pth` | 44,962,313 | `0474f612a62eba43b6051f7c0fee1b7dee480e201875a1c3be1155ef2f1a9935` |
| `Cifar100_Normal_Model.pth` | 44,961,249 | `7371ccb924d69da2d12906554a0fd7f0d3d73c6acb5ac78fc726a3f6a8ce7fcc` |
| `loss0.026.pth` | 154,716,830 | `2354ac1466ccc4e43dc2413a0daee07e4fe523bf28d7fec22c1d0dc6156560ac` |

Place `loss0.026.pth` in this folder. It is the CIFAR-100 diffusion prior loaded by `inversefed/reconstruction_algorithms.py`.

Place the three `Cifar100_*.pth` files in `../model/`:

- `Cifar100_Normal_Model.pth`: required before Stage 1, and used by later stages.
- `Cifar100_FK_0_Malicious_Model.pth`: Stage 1 output and Stage 2/3 input.
- `Cifar100_FK_0_MKrum_Poisoning_Model.pth`: Stage 2 output and Stage 3 input.

The duplicate copies originally stored in `trainedmodel/Cifar/` have identical SHA-256 hashes and are unnecessary in the source repository.

For the optional ImageNet branch, source defaults refer to `256x256_diffusion.pt` and `256x256_classifier.pt` in this folder. These were not in the supplied materials. See [Guided Diffusion](https://github.com/openai/guided-diffusion) for upstream models and configuration details; compatibility with this snapshot's ImageNet branch has not been validated.

To distribute your own weights, publish them separately using a GitHub Release or another download host and add the resulting URLs here. Keep hashes alongside the downloads. The 154,716,830-byte diffusion checkpoint exceeds GitHub's regular Git single-file limit; see [GitHub's large-file documentation](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).
