from dataclasses import asdict
import tyro
import torch

from nntool.slurm import slurm_function
from nntool.wandb import init_wandb
from accelerate import (
    Accelerator,
    DataLoaderConfiguration,
    DistributedDataParallelKwargs,
)
from accelerate.utils import set_seed, ProjectConfiguration
from src.model import MultiScaleConv3dEPBModel, MultiScaleAtomicEPBModel
from src.trainer import (
    EPB3dEnergyTrainer,
    Bf16EPB3dEnergyTrainer,
    Bf16VoxelEPB3dEnergyTrainer,
    AccelerateVoxelEPB3dEnergyTrainer,
    AccelerateAtomicEPB3dEnergyTrainer,
)
from configs.epb_3d.config import ConfiguredExperimentConfig, ExperimentConfig


@slurm_function
def main(args: ExperimentConfig):
    # use the same seed for all processes
    set_seed(args.seed)
    accelerator = Accelerator(
        dataloader_config=DataLoaderConfiguration(
            split_batches=args.trainer.split_batches
        ),
        project_config=ProjectConfiguration(
            args.trainer.output_folder, automatic_checkpoint_naming=True, total_limit=10
        ),
        gradient_accumulation_steps=args.trainer.gradient_accumulation_steps,
        mixed_precision="bf16" if args.trainer.do_bf16_training else "no",
        kwargs_handlers=[
            DistributedDataParallelKwargs(
                find_unused_parameters=args.trainer.find_unused_parameters
            )
        ],
    )

    if accelerator.is_main_process:
        init_wandb(args.wandb, asdict(args))

    if args.model_cls == "atomic_based":
        energy_model = MultiScaleAtomicEPBModel(args.atomic_energy_model)
    else:
        energy_model = MultiScaleConv3dEPBModel(args.energy_model)
    torch.save(
        energy_model.state_dict(),
        f"{args.trainer.output_folder}/model_init_weight_rank{accelerator.process_index}.pt",
    )

    TRAINER_CLS = {
        "patch_based": (
            Bf16EPB3dEnergyTrainer
            if args.trainer.do_bf16_training
            else EPB3dEnergyTrainer
        ),
        "voxel_based": AccelerateVoxelEPB3dEnergyTrainer,
        "atomic_based": AccelerateAtomicEPB3dEnergyTrainer,
    }
    trainer_cls = TRAINER_CLS[args.model_cls]
    trainer = trainer_cls(
        accelerator,
        energy_model,
        args.seed,
        args=args.trainer,
        has_wandb_writer=True if accelerator.is_main_process else False,
    )
    if args.resume_trainer:
        state_pt = torch.load(args.trainer_ckpt_path, map_location=trainer.device)
        trainer.load_state(state_pt)

    trainer.train()


if __name__ == "__main__":
    args: ExperimentConfig = tyro.parse(ConfiguredExperimentConfig)
    main(args.slurm)(args)
