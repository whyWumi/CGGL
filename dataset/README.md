# Local datasets

Dataset contents are ignored by Git. The CIFAR-100 wrapper in `dataset.py` uses torchvision to download into `Cifar100/train/` and `Cifar100/test/` below this directory.

The separately configured inversefed loaders use the same dataset root but can use a different internal layout. `CGGL_DATA_DIR` overrides the default root. Its optional `myimagenet` branch expects `myimage/` below that root; a complete ImageNet or LDCT reproduction setup is not included in this snapshot.
