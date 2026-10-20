# Third-party notices and provenance

The supplied project is a local code snapshot without upstream commit identifiers or a project-wide license. The list below records identifiable dependencies and source indicators; it is not a file-by-file authorship determination.

| Local component | Upstream or source indicator | License record |
| --- | --- | --- |
| `inversefed/` | [JonasGeiping/invertinggradients](https://github.com/JonasGeiping/invertinggradients), modified locally | [Upstream MIT license](licenses/invertinggradients-LICENSE) |
| `guided_diffusion/` | [openai/guided-diffusion](https://github.com/openai/guided-diffusion) | [Upstream MIT license](licenses/guided-diffusion-LICENSE) |
| `simpleAICV/` | [zgcr/SimpleAICV_pytorch_training_examples](https://github.com/zgcr/SimpleAICV_pytorch_training_examples), subset with local changes | [Upstream MIT license](licenses/SimpleAICV-LICENSE) |
| FL/poisoning framework | Original directory `PGLC-main`; original README title: “Extracting Private Training Data in Federated Learning from Clients” | No corresponding license or upstream URL was included in the supplied files |
| `inversefed/data/data.py` | Existing file header refers to [PyTorch examples](https://github.com/pytorch/examples/blob/master/super_resolution/data.py) | Existing attribution retained |

The three MIT license texts were retrieved from the identified upstream repositories on 2026-10-20. They record those upstream licenses; the precise revisions and all local modifications remain unidentified. Existing per-file source references and copyright notices have been retained.

The original README listed the PGLC title above and three `.ipynb` entry points. This snapshot contains the corresponding `.py` files, and the public README now describes those actual files.

A PGLC reference associated with that title is [Extracting Private Training Data in Federated Learning from Clients](https://doi.org/10.1109/TIFS.2025.3558581), Jiaheng Wei et al., IEEE Transactions on Information Forensics and Security (2025). This title match is a provenance clue, not confirmation of the exact upstream code revision.

No new project-wide license is assigned by this cleanup. The repository owner should establish permission for the PGLC-derived portions and select a license for their own contributions before describing the whole project as licensed open source. The paper PDF, datasets, checkpoints, and generated images are not relicensed or bundled here.
