# BASALT Competition

This approach won second place in the [2021 NeurIPS MineRL BASALT competition](https://minerl.io/basalt/):

> The MineRL Benchmark for Agents that Solve Almost-Lifelike Tasks (MineRL BASALT) competition aims to promote research in the area of learning from human feedback, in order to enable agents that can pursue tasks that do not have crisp, easily defined reward functions.

This developed in collaboration with [Divyansh Garg](https://github.com/Div99). Our general approach was to use [IQ-Learn](https://arxiv.org/abs/2106.12142) for online imitation learning. It is implemented using PyTorch.

## Citation
If you use this repo in your research, please consider citing the IQ-Learn paper as follows:

```
@article{
    title={IQ-Learn: Inverse soft-Q Learning for Imitation},
    author={Divyansh Garg, Shuvam Chakraborty, Chris Cundy, Jiaming Song, Stefano Ermon},
    year={2021},
    eprint={2106.12142},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

## Contents
### Environments
[MineRL](https://www.minerl.io)
- MineRLNavigateDense-v0
- MineRLNavigateExtremeDense-v0
- MineRLTreechop-v0
- MineRLBasaltFindCave-v0
- MineRLBasaltMakeWaterfall-v0
- MineRLBasaltCreateVillageAnimalPen-v0
- MineRLBasaltBuildVillageHouse-v0

### Algorithms
- [IQ-Learn](https://arxiv.org/abs/2106.12142) (Online and Offline)
- [SQIL](https://arxiv.org/abs/1905.11108) (Online and Offline)
- Behavioral Cloning

### Other Elements
- [Intrinsic Curiosity Module](https://pathak22.github.io/noreward-rl/)
- [Data Regularized Q (DRQ)](https://sites.google.com/view/data-regularized-q)

## Setup
### Options
- Locally: Package dependencies can be found in `environment.yml`, which can be set up with conda.
- Docker: you can generate a Docker image with `utility/docker_build.sh`
- Google Colab: You can run training via the `utility/colab.ipynb` file. It expects that you have this repository in a google drive folder

### Other Requirements
- MineRL: To set up MineRL, follow the setup instructions [here](https://minerl.readthedocs.io/en/latest/tutorials/index.html).
- WandB: We use [Weights & Biases](https://wandb.ai/) to track training metrics. You'll need to set up an account and log in when running training.

### Datasets
- Follow the instructions [here](https://minerl.readthedocs.io/en/latest/tutorials/data_sampling.html#downloading-the-minerl-dataset-with-minerl-data-download) to download and set up the BASALT competition datasets.

## Training
- To train locally, run `python train_submission_code.py`. Helpful flags include `--virtual-display-false`, `--debug-env`, and `--wandb-false`.
- Explore the config files in `conf/` to see the parameters available for modification and their default values. These can be overridden with additional arguments, e.g. `env=waterfall method.training_steps=100000`

## Evaluation
- Use `generate_trajectory.py` to download a model from wandb and generate trajectories.

## Testing
- Tests are implemented with `pytest`.

## License

Please see the [LICENSE](LICENSE.pdf) for the licensing terms for this code.
