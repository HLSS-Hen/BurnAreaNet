# BurnAreaNet
This repository contains the code for training and testing BurnAreaNet, as described in the paper:

> **BurnAreaNet: A Deep Learning Method for Estimating Total Body Surface Area from 2D Masks**

## Overview
The code in this repository is designed to reproduce the experiments and results from our paper on estimating the Total Body Surface Area (TBSA) affected by burns using deep learning and 2D masks.

## Getting Started
### Prerequisites
- Python 3.12
- Required Python packages (usually listed in ``requirements.txt``)

### Model Weights
Download the pre-trained model weights from Hugging Face:
[model.safetensors](https://huggingface.co/HLSS/BurnAreaNet/resolve/main/model.safetensors)

### Dataset
The dataset used for training and evaluation can be found on Hugging Face:

[Hugging Face Dataset: MassHumanBurns](https://huggingface.co/datasets/HLSS/MassHumanBurns)

For more information on how the dataset was constructed, please refer to the official builder tool:

[MassHumanBurns Dataset Builder](https://github.com/HLSS-Hen/MassHumanBurns_Builder)

## Demo
We provide an additional demo script in the ``demo/`` folder. **Please note that this demo is intended for demonstration purposes only.** It relies on a vanilla SAM2 model for mask generation, which may not produce perfect segmentations on our sample images and will therefore affect the final TBSA estimation accuracy.

## License
This project is licensed under the Apache 2.0 License.

## Citation
```bibtex
@article{burnareanet,
  title={BurnAreaNet: A Deep Learning Method for Estimating Total Body Surface Area from 2D Masks},
  author={Hao Wang, Kaize Zheng, Shuaidan Zeng, Yanyan Liang, Zhu Xiong},
  year={2026}
}
```
