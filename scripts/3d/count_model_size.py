import tyro

from torchinfo import summary
from src.model import MultiScaleConv3dEPBModel, MultiScaleAtomicEPBModel
from configs.epb_3d.config import ConfiguredExperimentConfig, ExperimentConfig


def main(args: ExperimentConfig):
    if args.model_cls == "atomic_based":
        energy_model = MultiScaleAtomicEPBModel(args.atomic_energy_model)
    else:
        energy_model = MultiScaleConv3dEPBModel(args.energy_model)

    summary(energy_model)


if __name__ == "__main__":
    args: ExperimentConfig = tyro.parse(ConfiguredExperimentConfig)
    main(args)
