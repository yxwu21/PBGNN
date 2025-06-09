from dataclasses import replace
from .experiment_config import ExperimentConfig, EvalExperimentConfig
from . import model_atomic
from . import trainer_atomic
from . import misc

experiments = dict(
    # small molecule with lset and neighbor list processed by dataloader
    distributed_atomic_all_atoms_grid35_small_mol_wo_lset_fully_coverage_medium_dataloader=ExperimentConfig(
        model_cls="atomic_based",
        atomic_energy_model=model_atomic.model_small,
        trainer=trainer_atomic.distributed_atomic_model_grid35_all_atoms_sparse_small_mol_fusion_fully_coverage_trainer_medium,
        wandb=misc.wandb,
        slurm=misc.distributed_slurm,
    ),
    debug_distributed_atomic_all_atoms_grid35_small_mol_wo_lset_fully_coverage_medium_dataloader=ExperimentConfig(
        model_cls="atomic_based",
        atomic_energy_model=model_atomic.model_small,
        trainer=replace(
            trainer_atomic.distributed_atomic_model_grid35_all_atoms_sparse_small_mol_fusion_fully_coverage_trainer_medium,
            eval_batch_size=1,
        ),
        wandb=misc.wandb,
        slurm=misc.debug_slurm,
    ),
    # all in one with lset and neighbor list processed by dataloader
    distributed_atomic_all_atoms_grid35_all_in_one_wo_lset_fully_coverage_medium_dataloader=ExperimentConfig(
        model_cls="atomic_based",
        atomic_energy_model=model_atomic.model_medium,
        trainer=trainer_atomic.distributed_atomic_model_grid35_all_atoms_sparse_all_in_one_fusion_fully_coverage_trainer_medium,
        wandb=misc.wandb,
        slurm=misc.distributed_slurm,
    ),
    distributed_atomic_all_atoms_grid35_all_in_one_with_lset_fully_coverage_medium_dataloader=ExperimentConfig(
        model_cls="atomic_based",
        atomic_energy_model=model_atomic.fusion_model_medium,
        trainer=trainer_atomic.distributed_atomic_model_grid35_all_atoms_sparse_all_in_one_fusion_fully_coverage_trainer_medium,
        wandb=misc.wandb,
        slurm=misc.distributed_slurm,
    ),
)
