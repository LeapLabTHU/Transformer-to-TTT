# Linearizing Vision Transformers with Test-Time Training

[![arXiv](https://img.shields.io/badge/arXiv-2605.02772-b31b1b.svg)](https://arxiv.org/abs/2605.02772)

Official implementation of **Linearizing Vision Transformers with Test-Time Training** (ICML 2026).

This repository contains two independent codebases:

- `T5/`: ImageNet classification experiments based on DeiT-style Vision Transformers.
- `DiT5/`: ImageNet class-conditional generation experiments based on DiT.

Each folder has its own environment and dependencies. Please enter the corresponding folder first, create the Conda environment, install the requirements, and then run the training command.

## Repository Structure

```text
Linearizing-Vision-Transformers-with-Test-Time-Training/
├── T5/       # T5 / DeiT-style image classification code
└── DiT5/     # DiT5 image generation code
```

## Data Preparation

Prepare ImageNet in the standard folder structure:

```text
/path/to/imagenet/
├── train/
└── val/
```

## Checkpoints

For `T5/`, download the DeiT-Tiny checkpoint used for weight inheritance:

- [DeiT-Tiny checkpoint](https://cloud.tsinghua.edu.cn/f/8efdb83ccae8422bb0ad/?dl=1)

For `DiT5/`, download the DiT-S/2 checkpoint used for weight inheritance:

- [DiT-S/2 checkpoint](https://cloud.tsinghua.edu.cn/f/ce8c6a6f96dc4d17aa80/?dl=1)

After downloading, replace the checkpoint path in the command with your local checkpoint path, for example:

```text
/path/to/deit_tiny.pth
/path/to/dit_s2_checkpoint.pt
```

## T<sup>5</sup>: Image Classification

Enter the `T5/` folder first:

```bash
cd T5
```

Create and activate the Conda environment:

```bash
conda create -n t5 python=3.10 -y
conda activate t5
pip install -r requirements.txt
```

Train `T5_tiny` on ImageNet:

```bash
torchrun --nproc_per_node=8 main.py \
  --model T5_tiny \
  --batch-size 256 \
  --data-path /path/to/imagenet \
  --output_dir /path/to/output/t5_tiny \
  --load-deit-to-ttt /path/to/deit_tiny.pth
```


## DiT<sup>5</sup>: Image Generation

Enter the `DiT5/` folder first:

```bash
cd DiT5
```

Create and activate the Conda environment:

```bash
conda create -n dit_ttt python=3.12 -y
conda activate dit_ttt
pip install -r requirements.txt --timeout 1000
```

Train `DiT-S/2` on ImageNet at 256×256 resolution:

```bash
torchrun --nproc_per_node=8 train.py \
  --model DiT-S/2 \
  --data-path /path/to/imagenet/train \
  --image-size 256 \
  --global-batch-size 256 \
  --results-dir /path/to/output/dit5_s2_256 \
  --load-softmax-to-ttt /path/to/dit_s2_checkpoint.pt
```



## Citation

If you find this repository useful, please consider citing our work:

```bibtex
@article{li2026linearizing,
  title={Linearizing Vision Transformer with Test-Time Training},
  author={Li, Yining and Han, Dongchen and Liu, Zeyu and Wang, Hanyi and Wang, Yulin and Huang, Gao},
  journal={arXiv preprint arXiv:2605.02772},
  year={2026}
}
```

## Acknowledgements

This code is developed on the top of [DeiT](https://github.com/facebookresearch/deit) and [DiT](https://github.com/facebookresearch/DiT). 

## Contact

If you have any questions, please feel free to contact the authors.

Yining Li: li-yn25@mails.tsinghua.edu.cn
