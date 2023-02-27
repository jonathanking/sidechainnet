from sidechainnet.research.cluster.train_pl import main
import unittest
import sys


def test_main(n):
    args = [
        "program",
        "--local_scn_path=/net/pulsar/home/koes/jok120/scnmin220915/scn_minimized.pkl",
        "--dynamic_batching=False", "--d_feedforward=256", "--d_in=1024",
        "--d_seq_embedding=16", "--dropout=0.11019872035522184", "--loss_name=mse",
        "--model=scn-trans-enc", "--n_heads=8", "--n_layers=13", "--opt_name=adam",
        "--seed=5262", "--opt_weight_decay=0.00005558883543826726", "--casp_version=12",
        "--casp_thinning=100", "--log_every_n_steps=1", "--auto_lr_find_custom=False",
        "--shuffle=false", "--num_sanity_val_steps=4", "--opt_patience=2",
        "--enable_checkpointing=True", "--early_stopping=True",
        "--early_stopping_patience=12", "--opt_lr_scheduling=noam",
        "--opt_lr=0.0201293750277862", "--opt_noam_lr_factor=0.1475447948767934",
        "--opt_n_warmup_steps=2000", "--batch_size=3", "--accumulate_grad_batches=4",
        "--opt_lr_scheduling_metric=train/gdc_all_step", "--auto_scale_batch_size=False",
        "--complete_structures_only=True", "--opt_begin_mse_openmm_step=2390",
        "--overfit_batches=0", "--viz_structures_every_n_steps=25",
        "--save_final_chkpt=./grateful-sweep-289v2.chkpt",
        "--loss_weight_mse=944.5612679661368", "--loss_weight_omm=58.76490492056114",
        "--name=gratefule-sweep-289v2", "--tags=retrain1004"
    ]
    if n == 1:
        args = [
        "program",
        "--local_scn_path=/net/pulsar/home/koes/jok120/scnmin220915/scn_minimized.pkl",
        "--dynamic_batching=False", "--d_feedforward=256", "--d_in=1024",
        "--d_seq_embedding=16", "--dropout=0.11019872035522184", "--loss_name=mse_openmm",
        "--model=scn-trans-enc", "--n_heads=8", "--n_layers=13", "--opt_name=adam",
        "--seed=5262", "--opt_weight_decay=0.00005558883543826726", "--casp_version=12",
        "--casp_thinning=100", "--log_every_n_steps=1", "--auto_lr_find_custom=False",
        "--shuffle=false", "--num_sanity_val_steps=4", "--opt_patience=2",
        "--enable_checkpointing=True", "--early_stopping=True",
        "--early_stopping_patience=12", "--opt_lr_scheduling=noam",
        "--opt_lr=0.0201293750277862", "--opt_noam_lr_factor=0.1475447948767934",
        "--opt_n_warmup_steps=2000", "--batch_size=3", "--accumulate_grad_batches=4",
        "--opt_lr_scheduling_metric=train/gdc_all_step", "--auto_scale_batch_size=False",
        "--complete_structures_only=True", "--opt_begin_mse_openmm_step=0",
        "--overfit_batches=0", "--viz_structures_every_n_steps=25",
        "--save_final_chkpt=./grateful-sweep-289v2.chkpt",
        "--loss_weight_mse=944.5612679661368", "--loss_weight_omm=58.76490492056114",
        "--name=gratefule-sweep-289v3", "--tags=retrain1004"
    ]
    with unittest.mock.patch("sys.argv", args):
        main()


if __name__ == "__main__":
    test_main(1)