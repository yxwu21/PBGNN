from nntool.slurm import SlurmConfig
from nntool.wandb import WandbConfig
from nntool.utils import get_output_path

output_path, current_time = get_output_path(append_date=True)

slurm = SlurmConfig(
    mode="slurm",
    partition="zhanglab.p",
    job_name="epb",
    output_folder=f"{output_path}/slurm",
    tasks_per_node=1,
    cpus_per_task=16,
    gpus_per_node=1,
    mem="256G",
    node_list="laniakea",
    use_distributed_env=False,
)

atomic_slurm = SlurmConfig(
    mode="slurm",
    partition="zhanglab.p",
    job_name="epb",
    output_folder=f"{output_path}/slurm",
    tasks_per_node=1,
    cpus_per_task=16,
    gpus_per_node=1,
    mem="128G",
    node_list="laniakea",
    use_distributed_env=False,
)

distributed_slurm = SlurmConfig(
    mode="slurm",
    partition="zhanglab.p",
    job_name="epb",
    output_folder=f"{output_path}/slurm",
    tasks_per_node=1,
    cpus_per_task=16,
    gpus_per_node=4,
    mem="384G",
    node_list="laniakea",
    use_distributed_env=True,
    processes_per_task=4,
    distributed_launch_command="accelerate launch --config_file configs/accelerate/accelerate.yaml --num_processes {num_processes} --num_machines {num_machines} --machine_rank {machine_rank} --main_process_ip {main_process_ip} --main_process_port {main_process_port} -m scripts.3d.train_3d_energy_distributed",
)

distributed_eval_slurm = SlurmConfig(
    mode="slurm",
    partition="zhanglab.p",
    job_name="epb",
    output_folder=f"{output_path}/slurm",
    tasks_per_node=1,
    cpus_per_task=24,
    gpus_per_node=4,
    mem="60G",
    node_list="laniakea",
    use_distributed_env=True,
    processes_per_task=4,
    distributed_launch_command="accelerate launch --config_file configs/accelerate/accelerate.yaml --num_processes {num_processes} --num_machines {num_machines} --machine_rank {machine_rank} --main_process_ip {main_process_ip} --main_process_port {main_process_port} -m scripts.3d.test_3d_energy_distributed",
)

debug_distributed_slurm = SlurmConfig(
    mode="debug",
    partition="zhanglab.p",
    job_name="epb",
    output_folder=f"{output_path}/slurm",
    tasks_per_node=1,
    cpus_per_task=8,
    gpus_per_node=4,
    mem="60G",
    node_list="laniakea",
    use_distributed_env=True,
    processes_per_task=4,
    distributed_launch_command="accelerate launch --config_file configs/accelerate/accelerate.yaml --num_processes {num_processes} --num_machines {num_machines} --machine_rank {machine_rank} --main_process_ip {main_process_ip} --main_process_port {main_process_port} -m scripts.3d.train_3d_energy_distributed",
)

debug_slurm = SlurmConfig(
    mode="debug",
    partition="zhanglab.p",
    job_name="epb",
    output_folder=f"{output_path}/slurm",
    tasks_per_node=1,
    cpus_per_task=8,
    gpus_per_node=1,
    mem="60G",
    node_list="laniakea",
    use_distributed_env=False,
)

wandb = WandbConfig(
    project="epb-surface",
    entity="junhaoliu17",
    name=f"3d_energy_train_{current_time}",
    api_key_config_file=".key",
)
