#!/bin/bash 
/scr/openfold/train_openfold.py /scr/alphafold_data/pdb_mmcif/pdb_files_for_scnmin/ /scr/scn_roda2/ /scr/alphafold_data/pdb_mmcif/mmcif_files_for_scn/ /scr/experiments/221212/out1 2021-10-10 --gpus=1 --checkpoint_every_epoch --obsolete_pdbs_file_path=/scr/alphafold_data/pdb_mmcif/obsolete.dat --config_preset=finetuning_sidechainnet --wandb --wandb_project=finetune-openfold-01 --wandb_entity=jonathanking --log_every_n_steps=1 --resume_from_ckpt=/scr/openfold/openfold/resources/openfold_params/initial_training.pt --resume_model_weights_only=True --train_epoch_len=30 --max_epochs=1 --debug --use_openmm=True --add_struct_metrics --openmm_weight=1e-7 --openmm_activation=None --use_scn_pdb_names --write_pdbs --write_pdbs_every_n_steps=3 --seed=0  --train_chain_data_cache_path=chain_data_cache.json --deepspeed_config_path=deepspeed_config.json --num_workers=1 --precision=bf16 --wandb_notes="Baseline. No template cache, 1 worker, deepspeed:bf16, no alignment index, no benchmark" --experiment=speed-test-00-baseline
/scr/openfold/train_openfold.py /scr/alphafold_data/pdb_mmcif/pdb_files_for_scnmin/ /scr/scn_roda2/ /scr/alphafold_data/pdb_mmcif/mmcif_files_for_scn/ /scr/experiments/221212/out1 2021-10-10 --gpus=1 --checkpoint_every_epoch --obsolete_pdbs_file_path=/scr/alphafold_data/pdb_mmcif/obsolete.dat --config_preset=finetuning_sidechainnet --wandb --wandb_project=finetune-openfold-01 --wandb_entity=jonathanking --log_every_n_steps=1 --resume_from_ckpt=/scr/openfold/openfold/resources/openfold_params/initial_training.pt --resume_model_weights_only=True --train_epoch_len=30 --max_epochs=1 --debug --use_openmm=True --add_struct_metrics --openmm_weight=1e-7 --openmm_activation=None --use_scn_pdb_names --write_pdbs --write_pdbs_every_n_steps=3 --seed=0  --train_chain_data_cache_path=chain_data_cache.json --deepspeed_config_path=deepspeed_config.json --num_workers=12 --precision=bf16 --wandb_notes="Baseline + 12 workers. No template cache, 12 workers, deepspeed:bf16, no alignment index, no benchmark" --experiment=speed-test-01-12workers
/scr/openfold/train_openfold.py /scr/alphafold_data/pdb_mmcif/pdb_files_for_scnmin/ /scr/scn_roda2/ /scr/alphafold_data/pdb_mmcif/mmcif_files_for_scn/ /scr/experiments/221212/out1 2021-10-10 --gpus=1 --checkpoint_every_epoch --obsolete_pdbs_file_path=/scr/alphafold_data/pdb_mmcif/obsolete.dat --config_preset=finetuning_sidechainnet --wandb --wandb_project=finetune-openfold-01 --wandb_entity=jonathanking --log_every_n_steps=1 --resume_from_ckpt=/scr/openfold/openfold/resources/openfold_params/initial_training.pt --resume_model_weights_only=True --train_epoch_len=30 --max_epochs=1 --debug --use_openmm=True --add_struct_metrics --openmm_weight=1e-7 --openmm_activation=None --use_scn_pdb_names --write_pdbs --write_pdbs_every_n_steps=3 --seed=0  --train_chain_data_cache_path=chain_data_cache.json --deepspeed_config_path=deepspeed_config.json --num_workers=16 --precision=bf16 --wandb_notes="Baseline + 16 workers. No template cache, 16 workers, deepspeed:bf16, no alignment index, no benchmark" --experiment=speed-test-02-16workers
/scr/openfold/train_openfold.py /scr/alphafold_data/pdb_mmcif/pdb_files_for_scnmin/ /scr/scn_roda2/ /scr/alphafold_data/pdb_mmcif/mmcif_files_for_scn/ /scr/experiments/221212/out1 2021-10-10 --gpus=1 --checkpoint_every_epoch --obsolete_pdbs_file_path=/scr/alphafold_data/pdb_mmcif/obsolete.dat --config_preset=finetuning_sidechainnet --wandb --wandb_project=finetune-openfold-01 --wandb_entity=jonathanking --log_every_n_steps=1 --resume_from_ckpt=/scr/openfold/openfold/resources/openfold_params/initial_training.pt --resume_model_weights_only=True --train_epoch_len=30 --max_epochs=1 --debug --use_openmm=True --add_struct_metrics --openmm_weight=1e-7 --openmm_activation=None --use_scn_pdb_names --write_pdbs --write_pdbs_every_n_steps=3 --seed=0  --train_chain_data_cache_path=chain_data_cache.json --deepspeed_config_path=deepspeed_config.json --num_workers=14 --precision=bf16 --wandb_notes="Baseline + 14 workers. No template cache, 14 workers, deepspeed:bf16, no alignment index, no benchmark" --experiment=speed-test-03-8workers
/scr/openfold/train_openfold.py /scr/alphafold_data/pdb_mmcif/pdb_files_for_scnmin/ /scr/scn_roda2/ /scr/alphafold_data/pdb_mmcif/mmcif_files_for_scn/ /scr/experiments/221212/out1 2021-10-10 --gpus=1 --checkpoint_every_epoch --obsolete_pdbs_file_path=/scr/alphafold_data/pdb_mmcif/obsolete.dat --config_preset=finetuning_sidechainnet --wandb --wandb_project=finetune-openfold-01 --wandb_entity=jonathanking --log_every_n_steps=1 --resume_from_ckpt=/scr/openfold/openfold/resources/openfold_params/initial_training.pt --resume_model_weights_only=True --train_epoch_len=30 --max_epochs=1 --debug --use_openmm=True --add_struct_metrics --openmm_weight=1e-7 --openmm_activation=None --use_scn_pdb_names --write_pdbs --write_pdbs_every_n_steps=3 --seed=0  --train_chain_data_cache_path=chain_data_cache.json --template_release_dates_cache_path=mmcif_cache.json --deepspeed_config_path=deepspeed_config.json --num_workers=1 --precision=bf16 --wandb_notes="Baseline + template cache. No template cache, 1 worker, deepspeed:bf16, no alignment index, no benchmark" --experiment=speed-test-04-tempcache
/scr/openfold/train_openfold.py /scr/alphafold_data/pdb_mmcif/pdb_files_for_scnmin/ /scr/scn_roda2/ /scr/alphafold_data/pdb_mmcif/mmcif_files_for_scn/ /scr/experiments/221212/out1 2021-10-10 --gpus=1 --checkpoint_every_epoch --obsolete_pdbs_file_path=/scr/alphafold_data/pdb_mmcif/obsolete.dat --config_preset=finetuning_sidechainnet --wandb --wandb_project=finetune-openfold-01 --wandb_entity=jonathanking --log_every_n_steps=1 --resume_from_ckpt=/scr/openfold/openfold/resources/openfold_params/initial_training.pt --resume_model_weights_only=True --train_epoch_len=30 --max_epochs=1 --debug --use_openmm=True --add_struct_metrics --openmm_weight=1e-7 --openmm_activation=None --use_scn_pdb_names --write_pdbs --write_pdbs_every_n_steps=3 --seed=0  --train_chain_data_cache_path=chain_data_cache.json --deepspeed_config_path=deepspeed_config.json --benchmark --num_workers=1 --precision=bf16 --wandb_notes="Baseline + benchmark. No template cache, 1 worker, deepspeed:bf16, no alignment index, yes benchmark" --experiment=speed-test-05-benchmark
/scr/openfold/train_openfold.py /scr/alphafold_data/pdb_mmcif/pdb_files_for_scnmin/ /scr/scn_roda2/ /scr/alphafold_data/pdb_mmcif/mmcif_files_for_scn/ /scr/experiments/221212/out1 2021-10-10 --gpus=1 --checkpoint_every_epoch --obsolete_pdbs_file_path=/scr/alphafold_data/pdb_mmcif/obsolete.dat --config_preset=finetuning_sidechainnet --wandb --wandb_project=finetune-openfold-01 --wandb_entity=jonathanking --log_every_n_steps=1 --resume_from_ckpt=/scr/openfold/openfold/resources/openfold_params/initial_training.pt --resume_model_weights_only=True --train_epoch_len=30 --max_epochs=1 --debug --use_openmm=True --add_struct_metrics --openmm_weight=1e-7 --openmm_activation=None --use_scn_pdb_names --write_pdbs --write_pdbs_every_n_steps=3 --seed=0  --train_chain_data_cache_path=chain_data_cache.json --deepspeed_config_path=deepspeed_config.json --num_workers=1 --precision=bf16 --alignment_index_path=/scr/dbs/super.index --wandb_notes="Baseline + alignment index. No template cache, 1 worker, deepspeed:bf16, yes alignment index, no benchmark" --experiment=speed-test-06-alnindex
/scr/openfold/train_openfold.py /scr/alphafold_data/pdb_mmcif/pdb_files_for_scnmin/ /scr/scn_roda2/ /scr/alphafold_data/pdb_mmcif/mmcif_files_for_scn/ /scr/experiments/221212/out1 2021-10-10 --gpus=1 --checkpoint_every_epoch --obsolete_pdbs_file_path=/scr/alphafold_data/pdb_mmcif/obsolete.dat --config_preset=finetuning_sidechainnet --wandb --wandb_project=finetune-openfold-01 --wandb_entity=jonathanking --log_every_n_steps=1 --resume_from_ckpt=/scr/openfold/openfold/resources/openfold_params/initial_training.pt --resume_model_weights_only=True --train_epoch_len=30 --max_epochs=1 --debug --use_openmm=True --add_struct_metrics --openmm_weight=1e-7 --openmm_activation=None --use_scn_pdb_names --write_pdbs --write_pdbs_every_n_steps=3 --seed=0  --train_chain_data_cache_path=chain_data_cache.json --template_release_dates_cache_path=mmcif_cache.json --deepspeed_config_path=deepspeed_config.json --benchmark --num_workers=1 --precision=bf16 --alignment_index_path=/scr/dbs/super.index --wandb_notes="Baseline + template cache + benchmark + alignment index. 1 worker, deepspeed:bf16" --experiment=speed-test-07-tempcache+benchmark+alnindex