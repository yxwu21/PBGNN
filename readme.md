# ❇️ PBGNN: End-to-End Modeling of Reaction Field Energy Using Data-Driven Geometric Graph Neural Networkss

![](assets/overview.jpg)

This repository implements **PBGNN**, including data preprocessing, loading, model training, and evaluation. The code is written in Python using PyTorch, enabling rapid prototyping and experimentation.


## Dependencies

### Requirements

- Python == 3.9
- PyTorch>=1.12.1
- Packages `pip install -r requirements.txt`

## Datasets

All datasets used for training and validation are located in the `datasets` folder.

### Preprocessing

One should use AMBER/PBSA to produce the EPB file and atom-wise potential file for each molecule. Once these files are produced,
preprocessing can be conducted via `python -m scripts.3d.prepare_3d_energy_sparse_dataset`.

To see the detailed configurations, please run

```bash
python -m scripts.3d.prepare_3d_energy_sparse_dataset -h

# output:
usage: prepare_3d_energy_sparse_dataset.py [-h] {amber_pbsa_epb_sparse_datset,pbsmall_epb_sparse_datset}

╭─ options ────────────────────────────────────────────────╮
│ -h, --help        show this help message and exit        │
╰──────────────────────────────────────────────────────────╯
╭─ subcommands ────────────────────────────────────────────╮
│ {amber_pbsa_epb_sparse_datset,pbsmall_epb_sparse_datset} │
│     amber_pbsa_epb_sparse_datset                         │
│     pbsmall_epb_sparse_datset                            │
╰──────────────────────────────────────────────────────────╯
```

To check the provided configuration for each subcommand, one can run the following command (using `amber_pbsa_epb_sparse_datset` as an example):

```bash
python -m scripts.3d.prepare_3d_energy_sparse_dataset amber_pbsa_epb_sparse_datset -h
```

### Preprocessed Data

To enable better reproducibility, we have provided preprocessed data files for the AMBER/PBSA and PBSMALL datasets. You can access the preprocessed datasets via [Zenodo](https://doi.org/10.5281/zenodo.15620599).

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
python -m scripts.3d.train_3d_energy_distributed distributed_atomic_all_atoms_grid35_all_in_one_with_lset_fully_coverage_medium_dataloader -h
```
