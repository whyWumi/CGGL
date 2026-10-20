# CGGL 代码说明

论文：**CGGL: A client-side generative gradient leakage attack with double diffusion prior**，*Information Fusion* 123 (2025), 103292。  
[论文链接](https://doi.org/10.1016/j.inffus.2025.103292) · [English README](README.md)

本项目研究联邦学习中结合客户端投毒与双扩散先验的梯度泄露攻击。

当前发布内容是整理后的研究代码快照，主要入口配置为 CIFAR-100。模型权重、数据集、日志和历史生成图没有包含在代码仓库中。完整论文实验尚未重新验证，原代码与论文的参数差异见 [复现说明](docs/REPRODUCIBILITY.md)。

## 安装与准备

在仓库根目录执行：

```bash
conda env create -f environment.yml
conda activate cggl
```

版本根据原环境整理：Python 3.9、PyTorch 2.0.0、torchvision 0.15.1。实验包含显式 CUDA 调用，需要兼容的 NVIDIA GPU 和 MPI 运行时。新的环境安装方案尚未在独立 GPU 环境验证。

将扩散模型 `loss0.026.pth` 放入 `checkpoints/`，将三个 CIFAR-100 联邦学习模型放入 `model/`。没有现成阶段模型时，应按顺序运行；第一个阶段本身仍需要已有的正常模型。当前正常模型训练代码是注释状态。权重名称、用途和 SHA-256 校验值见 [权重说明](checkpoints/README.md)。

```bash
python scripts/check_environment.py --stage invert --runtime
python run.py craft
python run.py poison
python run.py invert
```

启动器会创建输出目录并在仓库根目录运行原始脚本。CIFAR-100 数据默认自动下载到 `dataset/`。数据和扩散模型路径可通过 `cggl_paths.py` 中列出的环境变量覆盖。

## 使用范围

- 当前反演代码多处固定使用 8 张图像，修改 `--num_images` 不能独立完成批大小切换。
- ImageNet 分支仍有原始实验假设；LDCT 的完整入口、配置和数据处理链未包含在提供的材料中。
- 论文与代码的客户端数量、重建步数、种子数量等存在差异，整理时保留了原始数值。
- 已完成的源码检查不能代替 GPU 实验和论文数值复现。

保留了原有依赖库和实验分支；移除了发布版中的 IDE 配置、Python 缓存、历史输出和重复权重。第三方来源与许可证见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。原项目未提供整体许可证，整理过程未代替作者选择授权方式。
