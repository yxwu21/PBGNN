from typing import Union
import tyro
import joblib

from dataclasses import asdict
from nntool.slurm import slurm_function
from nntool.wandb import init_wandb
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import set_seed, ProjectConfiguration
from safetensors.torch import load
from src.model import MultiScaleConv3dEPBModel, MultiScaleAtomicEPBModel
from src.trainer import (
    EPB3dEnergyTrainer,
    Bf16EPB3dEnergyTrainer,
    AccelerateVoxelEPB3dEnergyTrainer,
    AccelerateAtomicEPB3dEnergyTrainer,
)
from configs.epb_3d.config import (
    ConfiguredEvalExperimentConfig,
    EvalExperimentConfig,
    ExperimentConfig,
)


@slurm_function
def main(args: Union[EvalExperimentConfig, ExperimentConfig]):
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
    )

    if accelerator.is_main_process:
        init_wandb(args.wandb, asdict(args))

    # reload model checkpoint
    if args.model_cls == "atomic_based":
        energy_model = MultiScaleAtomicEPBModel(args.atomic_energy_model)
    else:
        energy_model = MultiScaleConv3dEPBModel(args.energy_model)
    with open(args.model_ckpt_path, "rb") as f:
        data = f.read()
    energy_model.load_state_dict(load(data))

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

    # eval the model checkpoint
    scores, eval_output = trainer.eval_full_converage(trainer.eval_dl)
    trainer.log(scores, section="eval")
    trainer.accelerator.print("eval score:", scores)
    scores, test_output = trainer.eval_full_converage(trainer.test_dl)
    trainer.log(scores, section="test")
    trainer.accelerator.print("test score:", scores)

    # save ouputs
    outputs = {
        "eval": eval_output,
        "test": test_output,
    }
    joblib.dump(outputs, f"{trainer.output_folder}/epb_outputs.joblib")


if __name__ == "__main__":
    args: EvalExperimentConfig = tyro.parse(ConfiguredEvalExperimentConfig)
    main(args.slurm)(args)
