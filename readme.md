# ❇️ PBGNN: End-to-End Modeling of Reaction Field Energy Using Data-Driven Geometric Graph Neural Networkss

![](assets/overview.jpg)

This repository implements **PBGNN**, including data preprocessing, loading, model training, and evaluation. The code is written in Python using PyTorch, enabling rapid prototyping and experimentation.


## Dependencies

### Requirements

- Python >= 3.9
- Install the `slurm` plugin via `pip install wheel/nntool-1.6.2-py3-none-any.whl`
- Install the `pbgnn` package via `pip install -e .`
- Set up the `wandb` and `slurm` fields in the `env.toml` file

## Datasets

To enable better reproducibility, we have provided preprocessed data files for the AMBER/PBSA and PBSMALL datasets. You can access the preprocessed datasets via [Zenodo](https://doi.org/10.5281/zenodo.15867553).

## Training and Testing

To train and test PBGNN on the AMBER/PBSA dataset, run:

```bash
# training
sh scripts/3d/bash/train.sh

# testing
sh scripts/3d/bash/test.sh
```
One is able to modify the predefined configurations by running

```bash
python -m scripts.3d.train_3d_energy_distributed amber_pbsa -h
```

All arguments can be overrided by passing new argument values, for example,

```bash
python -m scripts.3d.train_3d_energy_distributed amber_pbsa --trainer.train-dataset-extra-config.neighbor-list-cutoff 15 --trainer.eval-dataset-extra-config.neighbor-list-cutoff 15 --trainer.train-num-steps 32000
```