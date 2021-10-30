# Imitation Learning

This repository is aims to be a flexible, extensible framework for imitation learning. It is implemented in pytorch.


## Environments

[MineRL](https://www.minerl.io)
- MineRLNavigateDense-v0
- MineRLNavigateExtremeDense-v0
- MineRLTreechop-v0
- MineRLBasaltFindCave-v0
- MineRLBasaltMakeWaterfall-v0
- MineRLBasaltCreateVillageAnimalPen-v0
- MineRLBasaltBuildVillageHouse-v0

## Algorithms
- IQ-Learn [1]
- SQIL [2]
- Behavioral Cloning

## Other References
- Intrinsic Curiosity Module [3]
- Data Regularized Q (DRQ) [4]

# Setup
- wandb

## Installation

## Downloading Datasets

# Training
`python train_submission_code.py`
- Explore the config files in conf/ to see the parameters available for modification and their default values.

# Evaluation

# Contributing
- pytest
- debug flags

# References
[1] @article{
    title={IQ-Learn: Inverse soft-Q Learning for Imitation},
    author={Divyansh Garg, Shuvam Chakraborty, Chris Cundy, Jiaming Song, Stefano Ermon},
    year={2021},
    eprint={2106.12142},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}

[2] @article{
    title={SQIL: Imitation Learning via Reinforcement Learning with Sparse Rewards},
    author={Siddharth Reddy, Anca D. Dragan, Sergey Levine},
    year={2019},
    eprint={1905.11108},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}

[3] @inproceedings{pathakICMl17curiosity,
    Author = {Pathak, Deepak and Agrawal, Pulkit and
              Efros, Alexei A. and Darrell, Trevor},
    Title = {Curiosity-driven Exploration by Self-supervised Prediction},
    Booktitle = {International Conference on Machine Learning ({ICML})},
    Year = {2017}
}

[4] @article{kostrikov2020image,
    title={Image Augmentation Is All You Need: Regularizing Deep Reinforcement Learning from Pixels},
    author={Ilya Kostrikov and Denis Yarats and Rob Fergus},
    year={2020},
    eprint={2004.13649},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
