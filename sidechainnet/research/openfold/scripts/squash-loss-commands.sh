#!/bin/bash
/scr/openfold/train_openfold.py /scr/alphafold_data/pdb_mmcif/pdb_files_for_scnmin/ /scr/scn_roda2/ /scr/alphafold_data/pdb_mmcif/mmcif_files_for_scn/ /scr/experiments/221212/out1 2021-10-10 --gpus=1 --deepspeed_config_path=deepspeed_config.json --checkpoint_every_epoch --train_chain_data_cache_path=chain_data_cache_scn2.json --obsolete_pdbs_file_path=/scr/alphafold_data/pdb_mmcif/obsolete.dat --config_preset=finetuning_sidechainnet --wandb --wandb_project=finetune-openfold-01 --wandb_entity=jonathanking --benchmark --num_workers=12 --log_every_n_steps=1 --precision=bf16 --resume_from_ckpt=/scr/openfold/openfold/resources/openfold_params/initial_training.pt --resume_model_weights_only=True --train_epoch_len=10 --max_epochs=60 --debug --use_openmm=True --add_struct_metrics --openmm_weight=1 --openmm_activation=None --use_scn_pdb_names --write_pdbs --write_pdbs_every_n_steps=1 --seed=0 --wandb_note="Squashed loss, C=100, weight=1" --overfit_single_batch --experiment=ftof-overfit-10-SL100 --openmm_squashed_loss=True --openmm_squashed_loss_factor=100
/scr/openfold/train_openfold.py /scr/alphafold_data/pdb_mmcif/pdb_files_for_scnmin/ /scr/scn_roda2/ /scr/alphafold_data/pdb_mmcif/mmcif_files_for_scn/ /scr/experiments/221212/out1 2021-10-10 --gpus=1 --deepspeed_config_path=deepspeed_config.json --checkpoint_every_epoch --train_chain_data_cache_path=chain_data_cache_scn2.json --obsolete_pdbs_file_path=/scr/alphafold_data/pdb_mmcif/obsolete.dat --config_preset=finetuning_sidechainnet --wandb --wandb_project=finetune-openfold-01 --wandb_entity=jonathanking --benchmark --num_workers=12 --log_every_n_steps=1 --precision=bf16 --resume_from_ckpt=/scr/openfold/openfold/resources/openfold_params/initial_training.pt --resume_model_weights_only=True --train_epoch_len=10 --max_epochs=60 --debug --use_openmm=True --add_struct_metrics --openmm_weight=1 --openmm_activation=None --use_scn_pdb_names --write_pdbs --write_pdbs_every_n_steps=1 --seed=0 --wandb_note="Squashed loss, C=200, weight=1" --overfit_single_batch --experiment=ftof-overfit-11-SL200 --openmm_squashed_loss=True --openmm_squashed_loss_factor=200
/scr/openfold/train_openfold.py /scr/alphafold_data/pdb_mmcif/pdb_files_for_scnmin/ /scr/scn_roda2/ /scr/alphafold_data/pdb_mmcif/mmcif_files_for_scn/ /scr/experiments/221212/out1 2021-10-10 --gpus=1 --deepspeed_config_path=deepspeed_config.json --checkpoint_every_epoch --train_chain_data_cache_path=chain_data_cache_scn2.json --obsolete_pdbs_file_path=/scr/alphafold_data/pdb_mmcif/obsolete.dat --config_preset=finetuning_sidechainnet --wandb --wandb_project=finetune-openfold-01 --wandb_entity=jonathanking --benchmark --num_workers=12 --log_every_n_steps=1 --precision=bf16 --resume_from_ckpt=/scr/openfold/openfold/resources/openfold_params/initial_training.pt --resume_model_weights_only=True --train_epoch_len=10 --max_epochs=60 --debug --use_openmm=True --add_struct_metrics --openmm_weight=1 --openmm_activation=None --use_scn_pdb_names --write_pdbs --write_pdbs_every_n_steps=1 --seed=0 --wandb_note="Squashed loss, C=100, weight=1, FCV=10" --overfit_single_batch --experiment=ftof-overfit-12-SL100-FCV10 --openmm_squashed_loss=True --openmm_squashed_loss_factor=100 --force_clipping_val=10
/scr/openfold/train_openfold.py /scr/alphafold_data/pdb_mmcif/pdb_files_for_scnmin/ /scr/scn_roda2/ /scr/alphafold_data/pdb_mmcif/mmcif_files_for_scn/ /scr/experiments/221212/out1 2021-10-10 --gpus=1 --deepspeed_config_path=deepspeed_config.json --checkpoint_every_epoch --train_chain_data_cache_path=chain_data_cache_scn2.json --obsolete_pdbs_file_path=/scr/alphafold_data/pdb_mmcif/obsolete.dat --config_preset=finetuning_sidechainnet --wandb --wandb_project=finetune-openfold-01 --wandb_entity=jonathanking --benchmark --num_workers=12 --log_every_n_steps=1 --precision=bf16 --resume_from_ckpt=/scr/openfold/openfold/resources/openfold_params/initial_training.pt --resume_model_weights_only=True --train_epoch_len=10 --max_epochs=60 --debug --use_openmm=True --add_struct_metrics --openmm_weight=1e-7 --openmm_activation=None --use_scn_pdb_names --write_pdbs --write_pdbs_every_n_steps=1 --seed=0 --wandb_note="UnSquashed loss, weight=1e-7, FCV=10" --overfit_single_batch --experiment=ftof-overfit-12-FCV10 --openmm_squashed_loss=False --openmm_squashed_loss_factor=100 --force_clipping_val=10