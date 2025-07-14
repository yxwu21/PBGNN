date_time=$(python -c "from nntool.utils import get_current_time; print(get_current_time())")
export OUTPUT_PATH=outputs/test_3d_energy_atomic/$date_time

python -m scripts.3d.test_3d_energy_distributed amber_pbsa \
    --model-ckpt-path outputs/train_3d_energy_atomic/10272024/223858/models/model_79/model.safetensors \
    --trainer.train-dataset-extra-config.neighbor-list-cutoff 15 \
    --trainer.eval-dataset-extra-config.neighbor-list-cutoff 15 \
    --slurm.distributed-launch-command "accelerate launch --config_file configs/accelerate/accelerate.yaml --num_processes {num_processes} --num_machines {num_machines} --machine_rank {machine_rank} --main_process_ip {main_process_ip} --main_process_port {main_process_port} -m scripts.3d.test_3d_energy_distributed"
