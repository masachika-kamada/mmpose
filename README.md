# MMPose

* Forked from [open-mmlab/mmpose](https://github.com/open-mmlab/mmpose)
* The demonstration programme has been improved by limiting it to 3D pose estimation only.

## Introduction

MMPose is an open-source toolbox for pose estimation based on PyTorch.
It is a part of the [OpenMMLab project](https://github.com/open-mmlab).

## Installation

MMPose depends on [PyTorch](https://pytorch.org/) and [MMCV](https://github.com/open-mmlab/mmcv).
Below are quick steps for installation.
Please refer to [install.md](docs/en/install.md) for detailed installation guide.

```shell
conda create -n openmmlab python=3.8 pytorch=1.10 cudatoolkit=11.3 torchvision -c pytorch -y
conda activate openmmlab
pip3 install openmim
mim install mmcv-full
git clone https://github.com/open-mmlab/mmpose.git
cd mmpose
pip3 install -e .
```

## Getting Started

Please see [get_started.md](docs/en/get_started.md) for the basic usage of MMPose.
