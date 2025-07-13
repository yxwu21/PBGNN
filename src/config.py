from dataclasses import dataclass
from typing import Literal


@dataclass
class RunConfig:
    # code running mode
    mode: Literal[
        "train", "finetune", "eval", "count", "plot", "process_data", "feature", "trace"
    ] = "train"

    # code running seed
    seed: int = 2024


@dataclass
class WandbConfig:
    # wandb run name
    name: str

    # project name in wandb
    project: str = "pbgnn"

    # wandb user name
    entity: str = ""

    # wandb api key
    api_key: str = ""
