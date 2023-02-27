#!/bin/bash
/scr/openfold/train_openfold.py /scr/alphafold_data/pdb_mmcif/pdb_files_for_scnmin/ /scr/scn_roda2/ /scr/alphafold_data/pdb_mmcif/mmcif_files_for_scn/ /scr/experiments/221212/out1 2021-10-10 --gpus=1 --deepspeed_config_path=deepspeed_config.json --checkpoint_every_epoch --train_chain_data_cache_path=chain_data_cache_scn2.json --obsolete_pdbs_file_path=/scr/alphafold_data/pdb_mmcif/obsolete.dat --config_preset=finetuning_sidechainnet --wandb --wandb_project=finetune-openfold-01 --wandb_entity=jonathanking --benchmark --num_workers=12 --log_every_n_steps=1 --precision=bf16 --resume_from_ckpt=/scr/openfold/openfold/resources/openfold_params/initial_training.pt --resume_model_weights_only=True --train_epoch_len=10 --max_epochs=60 --debug --use_openmm=True --add_struct_metrics --openmm_weight=1 --openmm_activation=None --use_scn_pdb_names --write_pdbs --write_pdbs_every_n_steps=1 --seed=0 --wandb_note="Modified_sigmoid 1" --overfit_single_batch --experiment="ftof-overfit-14-sig(5,1e6,300k,5)" --openmm_squashed_loss=False --openmm_squashed_loss_factor=200 --openmm_modified_sigmoid=5,1000000,300000,5
/scr/openfold/train_openfold.py /scr/alphafold_data/pdb_mmcif/pdb_files_for_scnmin/ /scr/scn_roda2/ /scr/alphafold_data/pdb_mmcif/mmcif_files_for_scn/ /scr/experiments/221212/out1 2021-10-10 --gpus=1 --deepspeed_config_path=deepspeed_config.json --checkpoint_every_epoch --train_chain_data_cache_path=chain_data_cache_scn2.json --obsolete_pdbs_file_path=/scr/alphafold_data/pdb_mmcif/obsolete.dat --config_preset=finetuning_sidechainnet --wandb --wandb_project=finetune-openfold-01 --wandb_entity=jonathanking --benchmark --num_workers=12 --log_every_n_steps=1 --precision=bf16 --resume_from_ckpt=/scr/openfold/openfold/resources/openfold_params/initial_training.pt --resume_model_weights_only=True --train_epoch_len=10 --max_epochs=60 --debug --use_openmm=True --add_struct_metrics --openmm_weight=1 --openmm_activation=None --use_scn_pdb_names --write_pdbs --write_pdbs_every_n_steps=1 --seed=0 --wandb_note="Modified_sigmoid 2" --overfit_single_batch --experiment="ftof-overfit-14-sig(4,-50k,74k,4)" --openmm_squashed_loss=False --openmm_squashed_loss_factor=200 --openmm_modified_sigmoid=4,-50000,74000,4
/scr/openfold/train_openfold.py /scr/alphafold_data/pdb_mmcif/pdb_files_for_scnmin/ /scr/scn_roda2/ /scr/alphafold_data/pdb_mmcif/mmcif_files_for_scn/ /scr/experiments/221212/out1 2021-10-10 --gpus=1 --deepspeed_config_path=deepspeed_config.json --checkpoint_every_epoch --train_chain_data_cache_path=chain_data_cache_scn2.json --obsolete_pdbs_file_path=/scr/alphafold_data/pdb_mmcif/obsolete.dat --config_preset=finetuning_sidechainnet --wandb --wandb_project=finetune-openfold-01 --wandb_entity=jonathanking --benchmark --num_workers=12 --log_every_n_steps=1 --precision=bf16 --resume_from_ckpt=/scr/openfold/openfold/resources/openfold_params/initial_training.pt --resume_model_weights_only=True --train_epoch_len=10 --max_epochs=60 --debug --use_openmm=True --add_struct_metrics --openmm_weight=1 --openmm_activation=None --use_scn_pdb_names --write_pdbs --write_pdbs_every_n_steps=1 --seed=0 --wandb_note="Modified_sigmoid 3" --overfit_single_batch --experiment="ftof-overfit-14-sig(5,174k,74k,1)" --openmm_squashed_loss=False --openmm_squashed_loss_factor=200 --openmm_modified_sigmoid=5,174000,74000,1
