date_time=$(python -c "from nntool.utils import get_current_time; print(get_current_time())")
export OUTPUT_PATH=outputs/train_3d_energy_atomic/$date_time

python -m scripts.3d.train_3d_energy_distributed amber_pbsa --trainer.train-dataset-extra-config.neighbor-list-cutoff 15 --trainer.eval-dataset-extra-config.neighbor-list-cutoff 15 --trainer.train-num-steps 32000
