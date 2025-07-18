from src.trainer import EPB3dEnergyTrainerConfig
from src.data import VoxelImage3dEnergyDatasetExtraConfig
from .misc import output_path

epb_model_psz64_ctx32_sparse_protein_complex_fully_coverage_rotation_augmented_trainer_medium = (
    EPB3dEnergyTrainerConfig(
        dataset_path="datasets/processed/new_full_3d_energy_sparse_v2/*/*/*/*_sparse.pkl.gz",
        pkl_filter="protein_protein_test",
        patch_size=64,
        output_folder=output_path,
        train_lr=5e-4,
        final_train_lr=1e-6,
        train_num_steps=8000,
        train_batch_size=32,
        eval_batch_size=1,
        save_and_eval_every=400,
        num_workers=8,
        probe_radius_upperbound=1.5,
        probe_radius_lowerbound=-5,
        do_bf16_training=True,
        gradient_accumulation_steps=2,
        do_random_crop=True,
        random_crop_atom_num=0.1,
        random_crop_interval=32,
        use_sparse_dataset=True,
        use_full_coverage_sparse_dataset=True,
        use_atomic_sparse_dataset=True,
        full_coverage_chunk_size=4,
        do_random_rotate=False,
        do_voxel_grids_shrinking=False,
        train_dataset_extra_config=VoxelImage3dEnergyDatasetExtraConfig(context_size=32),
        eval_dataset_extra_config=VoxelImage3dEnergyDatasetExtraConfig(context_size=32),
    )
)

epb_model_psz64_ctx32_sparse_all_in_one_fully_coverage_rotation_augmented_trainer_medium = (
    EPB3dEnergyTrainerConfig(
        dataset_path="datasets/processed/new_full_3d_energy_sparse_v2/*/*/*/*_sparse.pkl.gz",
        patch_size=64,
        output_folder=output_path,
        train_lr=5e-4,
        final_train_lr=1e-6,
        train_num_steps=8000,
        train_batch_size=32,
        eval_batch_size=1,
        save_and_eval_every=400,
        num_workers=8,
        probe_radius_upperbound=1.5,
        probe_radius_lowerbound=-5,
        do_bf16_training=True,
        gradient_accumulation_steps=2,
        do_random_crop=True,
        random_crop_atom_num=0.1,
        random_crop_interval=32,
        use_sparse_dataset=True,
        use_full_coverage_sparse_dataset=True,
        use_atomic_sparse_dataset=True,
        full_coverage_chunk_size=4,
        do_random_rotate=False,
        do_voxel_grids_shrinking=False,
        train_dataset_extra_config=VoxelImage3dEnergyDatasetExtraConfig(context_size=32),
        eval_dataset_extra_config=VoxelImage3dEnergyDatasetExtraConfig(context_size=32),
    )
)

epb_model_all_atoms_sparse_all_in_one_fully_coverage_rotation_augmented_trainer_medium = (
    EPB3dEnergyTrainerConfig(
        dataset_path="datasets/processed/new_full_3d_energy_sparse_v2/*/*/*/*_sparse.pkl.gz",
        patch_size=-1,
        output_folder=output_path,
        train_lr=5e-4,
        final_train_lr=1e-6,
        train_num_steps=8000,
        train_batch_size=32,
        eval_batch_size=1,
        save_and_eval_every=400,
        num_workers=8,
        probe_radius_upperbound=1.5,
        probe_radius_lowerbound=-5,
        do_bf16_training=True,
        gradient_accumulation_steps=2,
        do_random_crop=False,
        random_crop_atom_num=0.1,
        random_crop_interval=32,
        use_sparse_dataset=True,
        use_full_coverage_sparse_dataset=True,
        use_atomic_sparse_dataset=True,
        full_coverage_chunk_size=4,
        do_random_rotate=False,
        do_voxel_grids_shrinking=False,
        train_dataset_extra_config=VoxelImage3dEnergyDatasetExtraConfig(context_size=0),
        eval_dataset_extra_config=VoxelImage3dEnergyDatasetExtraConfig(context_size=0),
    )
)

distributed_epb_model_all_atoms_sparse_all_in_one_fully_coverage_rotation_augmented_trainer_medium = EPB3dEnergyTrainerConfig(
    dataset_path="datasets/processed/new_full_3d_energy_sparse_v2/*/*/*/*_sparse.pkl.gz",
    patch_size=-1,
    output_folder=output_path,
    train_lr=5e-4,
    final_train_lr=1e-6,
    train_num_steps=8000,
    train_batch_size=32,
    eval_batch_size=4,
    save_and_eval_every=400,
    num_workers=8,
    probe_radius_upperbound=1.5,
    probe_radius_lowerbound=-5,
    do_bf16_training=True,
    gradient_accumulation_steps=2,
    do_random_crop=False,
    random_crop_atom_num=0.1,
    random_crop_interval=32,
    use_sparse_dataset=True,
    use_full_coverage_sparse_dataset=True,
    use_atomic_sparse_dataset=True,
    full_coverage_chunk_size=4,
    do_random_rotate=False,
    do_voxel_grids_shrinking=False,
    train_dataset_extra_config=VoxelImage3dEnergyDatasetExtraConfig(context_size=0),
    eval_dataset_extra_config=VoxelImage3dEnergyDatasetExtraConfig(context_size=0),
)

distributed_atomic_model_all_atoms_sparse_all_in_one_fully_coverage_trainer_medium = (
    EPB3dEnergyTrainerConfig(
        dataset_path="datasets/processed/new_full_3d_energy_sparse_v2/*/*/*/*_sparse.pkl.gz",
        patch_size=-1,
        output_folder=output_path,
        train_lr=5e-4,
        final_train_lr=None,
        train_num_steps=16000,
        train_batch_size=4,
        eval_batch_size=4,
        save_and_eval_every=400,
        num_workers=8,
        probe_radius_upperbound=1.5,
        probe_radius_lowerbound=-5,
        do_bf16_training=True,
        gradient_accumulation_steps=2,
        do_random_crop=False,
        random_crop_atom_num=0.1,
        random_crop_interval=32,
        use_sparse_dataset=True,
        use_full_coverage_sparse_dataset=True,
        use_atomic_sparse_dataset=True,
        full_coverage_chunk_size=4,
        do_random_rotate=False,
        do_voxel_grids_shrinking=False,
        train_dataset_extra_config=VoxelImage3dEnergyDatasetExtraConfig(context_size=0),
        eval_dataset_extra_config=VoxelImage3dEnergyDatasetExtraConfig(context_size=0),
    )
)

distributed_atomic_model_all_atoms_sparse_small_mol_fully_coverage_trainer_medium = (
    EPB3dEnergyTrainerConfig(
        dataset_path="datasets/processed/small_mol_full_3d_energy_sparse/*/*/*/*_sparse.pkl.gz",
        patch_size=-1,
        output_folder=output_path,
        train_lr=5e-4,
        final_train_lr=1e-6,
        smooth_l1_loss_beta=1e-3,
        train_num_steps=16000,
        train_batch_size=4,
        eval_batch_size=4,
        save_and_eval_every=400,
        num_workers=8,
        probe_radius_upperbound=1.5,
        probe_radius_lowerbound=-5,
        do_bf16_training=True,
        gradient_accumulation_steps=2,
        do_random_crop=False,
        random_crop_atom_num=0.1,
        random_crop_interval=32,
        use_sparse_dataset=True,
        use_full_coverage_sparse_dataset=True,
        use_atomic_sparse_dataset=True,
        full_coverage_chunk_size=4,
        do_random_rotate=False,
        do_voxel_grids_shrinking=False,
        train_dataset_extra_config=VoxelImage3dEnergyDatasetExtraConfig(context_size=0),
        eval_dataset_extra_config=VoxelImage3dEnergyDatasetExtraConfig(context_size=0),
    )
)

distributed_atomic_model_grid35_all_atoms_sparse_all_in_one_fully_coverage_trainer_medium = EPB3dEnergyTrainerConfig(
    dataset_path="datasets/processed/new_full_3d_energy_sparse_v2/bench_full_abs_cpu_035/*/*/*_sparse.pkl.gz",
    patch_size=-1,
    output_folder=output_path,
    train_lr=5e-4,
    final_train_lr=None,
    train_num_steps=16000,
    train_batch_size=4,
    eval_batch_size=4,
    save_and_eval_every=400,
    num_workers=8,
    probe_radius_upperbound=1.5,
    probe_radius_lowerbound=-5,
    do_bf16_training=True,
    gradient_accumulation_steps=2,
    do_random_crop=False,
    random_crop_atom_num=0.1,
    random_crop_interval=32,
    use_sparse_dataset=True,
    use_full_coverage_sparse_dataset=True,
    use_atomic_sparse_dataset=True,
    full_coverage_chunk_size=4,
    do_random_rotate=False,
    do_voxel_grids_shrinking=False,
    train_dataset_extra_config=VoxelImage3dEnergyDatasetExtraConfig(context_size=0),
    eval_dataset_extra_config=VoxelImage3dEnergyDatasetExtraConfig(context_size=0),
)

amber_pbsa_atomic_trainer = EPB3dEnergyTrainerConfig(
    dataset_path="datasets/processed/new_full_3d_energy_sparse_v2/bench_full_abs_cpu_035/*/*/*_sparse.pkl.gz",
    patch_size=-1,
    output_folder=output_path,
    train_lr=5e-4,
    final_train_lr=None,
    train_num_steps=16000,
    train_batch_size=4,
    eval_batch_size=4,
    save_and_eval_every=400,
    num_workers=8,
    probe_radius_upperbound=1.5,
    probe_radius_lowerbound=-5,
    do_bf16_training=True,
    gradient_accumulation_steps=2,
    do_random_crop=False,
    random_crop_atom_num=0.1,
    random_crop_interval=32,
    use_sparse_dataset=True,
    use_full_coverage_sparse_dataset=True,
    use_atomic_sparse_dataset=True,
    full_coverage_chunk_size=4,
    do_random_rotate=False,
    do_voxel_grids_shrinking=False,
    train_dataset_extra_config=VoxelImage3dEnergyDatasetExtraConfig(
        context_size=0,
        process_atom_neighbor_pairs=True,
        neighbor_list_cutoff=10.0,
        natom_threshold_for_max_num_neighbors=30000,
        max_num_neighbors=16,
        # select_nearest_level_set=True,
        # select_nearest_level_set_k=6,
    ),
    eval_dataset_extra_config=VoxelImage3dEnergyDatasetExtraConfig(
        context_size=0,
        process_atom_neighbor_pairs=True,
        neighbor_list_cutoff=10.0,
        natom_threshold_for_max_num_neighbors=30000,
        max_num_neighbors=16,
        # select_nearest_level_set=True,
        # select_nearest_level_set_k=6,
    ),
)

distributed_atomic_model_grid35_all_atoms_sparse_small_mol_fully_coverage_trainer_medium = EPB3dEnergyTrainerConfig(
    dataset_path="datasets/processed/small_mol_full_3d_energy_sparse/bench_full_abs_cpu_035/*/*/*_sparse.pkl.gz",
    patch_size=-1,
    output_folder=output_path,
    train_lr=5e-4,
    final_train_lr=1e-6,
    smooth_l1_loss_beta=1e-3,
    train_num_steps=16000,
    train_batch_size=4,
    eval_batch_size=4,
    save_and_eval_every=400,
    num_workers=8,
    probe_radius_upperbound=1.5,
    probe_radius_lowerbound=-5,
    do_bf16_training=True,
    gradient_accumulation_steps=2,
    do_random_crop=False,
    random_crop_atom_num=0.1,
    random_crop_interval=32,
    use_sparse_dataset=True,
    use_full_coverage_sparse_dataset=True,
    use_atomic_sparse_dataset=True,
    full_coverage_chunk_size=4,
    do_random_rotate=False,
    do_voxel_grids_shrinking=False,
    train_dataset_extra_config=VoxelImage3dEnergyDatasetExtraConfig(context_size=0),
    eval_dataset_extra_config=VoxelImage3dEnergyDatasetExtraConfig(context_size=0),
)

pbsmall_variation_atomic_trainer = EPB3dEnergyTrainerConfig(
    dataset_path="datasets/processed/small_mol_full_3d_energy_sparse/bench_full_abs_cpu_035/*/*/*_sparse.pkl.gz",
    patch_size=-1,
    output_folder=output_path,
    train_lr=5e-4,
    final_train_lr=None,
    smooth_l1_loss_beta=1e-3,
    train_num_steps=16000,
    train_batch_size=32,
    eval_batch_size=4,
    save_and_eval_every=400,
    num_workers=8,
    probe_radius_upperbound=1.5,
    probe_radius_lowerbound=-5,
    do_bf16_training=True,
    gradient_accumulation_steps=1,
    find_unused_parameters=True,
    do_random_crop=False,
    random_crop_atom_num=0.1,
    random_crop_interval=32,
    use_sparse_dataset=True,
    use_full_coverage_sparse_dataset=True,
    use_atomic_sparse_dataset=True,
    full_coverage_chunk_size=4,
    do_random_rotate=False,
    do_voxel_grids_shrinking=False,
    train_dataset_extra_config=VoxelImage3dEnergyDatasetExtraConfig(
        context_size=0,
        process_atom_neighbor_pairs=True,
        neighbor_list_cutoff=30.0,
        natom_threshold_for_max_num_neighbors=30000,
        max_num_neighbors=16,
        select_nearest_level_set=True,
        select_nearest_level_set_k=6,
    ),
    eval_dataset_extra_config=VoxelImage3dEnergyDatasetExtraConfig(
        context_size=0,
        process_atom_neighbor_pairs=True,
        neighbor_list_cutoff=30.0,
        natom_threshold_for_max_num_neighbors=30000,
        max_num_neighbors=16,
        select_nearest_level_set=True,
        select_nearest_level_set_k=6,
    ),
)

pbsmall_atomic_trainer = EPB3dEnergyTrainerConfig(
    dataset_path="datasets/processed/small_mol_full_3d_energy_sparse/bench_full_abs_cpu_035/*/*/*_sparse.pkl.gz",
    patch_size=-1,
    output_folder=output_path,
    train_lr=5e-4,
    final_train_lr=None,
    smooth_l1_loss_beta=1e-3,
    train_num_steps=16000,
    train_batch_size=32,
    eval_batch_size=4,
    save_and_eval_every=400,
    num_workers=8,
    probe_radius_upperbound=1.5,
    probe_radius_lowerbound=-5,
    do_bf16_training=True,
    gradient_accumulation_steps=1,
    find_unused_parameters=True,
    do_random_crop=False,
    random_crop_atom_num=0.1,
    random_crop_interval=32,
    use_sparse_dataset=True,
    use_full_coverage_sparse_dataset=True,
    use_atomic_sparse_dataset=True,
    full_coverage_chunk_size=4,
    do_random_rotate=False,
    do_voxel_grids_shrinking=False,
    train_dataset_extra_config=VoxelImage3dEnergyDatasetExtraConfig(
        context_size=0,
        process_atom_neighbor_pairs=True,
        neighbor_list_cutoff=30.0,
        natom_threshold_for_max_num_neighbors=30000,
        max_num_neighbors=16,
        select_nearest_level_set=True,
        select_nearest_level_set_k=6,
        create_boundary_features=True,
        boundary_cutoff=1,
        boundary_unit=0.5,
    ),
    eval_dataset_extra_config=VoxelImage3dEnergyDatasetExtraConfig(
        context_size=0,
        process_atom_neighbor_pairs=True,
        neighbor_list_cutoff=30.0,
        natom_threshold_for_max_num_neighbors=30000,
        max_num_neighbors=16,
        select_nearest_level_set=True,
        select_nearest_level_set_k=6,
        create_boundary_features=True,
        boundary_cutoff=1,
        boundary_unit=0.5,
    ),
)
