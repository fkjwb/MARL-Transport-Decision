# 项目环境配置指南

本项目使用 Conda 进行依赖管理。我们提供了一个 `environment.yml` 文件，您可以利用它快速复现包含 Python 3.9、PyTorch (CUDA 12.4 兼容)、Torchaudio、Torchvision 以及 Jupyter 的开发环境。

## 1. 准备 Conda

在开始配置之前，请确保您的计算机上已经安装了 Conda 环境。
* 如果您还没有安装，推荐下载并安装轻量级的 [Miniconda](https://docs.anaconda.com/free/miniconda/index.html)，或者包含更多基础科学计算包的 [Anaconda](https://www.anaconda.com/download)。
* 安装完成后，您可以打开终端（Windows 推荐使用 Anaconda Prompt 或 PowerShell）并输入以下命令，以验证是否安装成功：

    ```bash
    conda --version
    ```

## 2. 使用环境文件创建环境

在包含 `environment.yml` 文件的项目根目录下打开终端，运行以下命令即可自动下载依赖并创建虚拟环境：

```bash
conda env create -f environment.yml
