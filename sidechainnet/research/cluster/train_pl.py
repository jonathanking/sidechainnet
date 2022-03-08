import argparse
import os
import multiprocessing as mp

import pytorch_lightning as pl
import pytorch_lightning.callbacks as callbacks
from pytorch_lightning.tuner.tuning import Tuner
import sidechainnet as scn
import torch
import torch.utils.data
from pl_bolts.callbacks import (BatchGradientVerificationCallback, ModuleDataMonitor)
from pytorch_lightning.loggers import WandbLogger
from sidechainnet.examples.LitSidechainTransformer import (LitSCNDataModule,
                                                           LitSidechainTransformer)

import wandb

torch.set_printoptions(precision=5, sci_mode=False)

# TODO: variable sq length https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html


def create_parser():
    """Create an argument parser for the program."""

    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    def my_bool(s):
        """Allow bools instead of pos/neg flags."""
        return s != 'False'

    parser = argparse.ArgumentParser()

    # Pytorch Lightning Trainer args (e.g. gpus, accumulate_grad_batches etc.)
    parser = pl.Trainer.add_argparse_args(parser)

    # Program level args
    program_args = parser.add_argument_group("Program")
    program_args.add_argument('--cluster',
                              '-c',
                              type=my_bool,
                              default="False",
                              help="Set of parameters to facilitate training on a remote"
                              " cluster. Limited I/O, etc.")
    program_args.add_argument('--model',
                              '-m',
                              choices=["scn-trans-enc"],
                              default="scn-trans-enc",
                              help="Model architecture name.")
    program_args.add_argument("--seed",
                              "-s",
                              type=int,
                              default=11_731,
                              help="The random number generator seed for numpy "
                              "and torch.")

    # Data arguments
    data_args = parser.add_argument_group("Data")
    data_args.add_argument("--casp_version",
                           choices=[7, 8, 9, 10, 11, 12],
                           type=int,
                           help="CASP Version for SidechainNet {7-12}.",
                           default=12)
    data_args.add_argument("--casp_thinning",
                           "--thinning",
                           choices=['30', '50', '70', '90', '100', 'debug'],
                           help="CASP thinning for SidechainNet.",
                           default='30')
    data_args.add_argument("--scn_data_file",
                           type=str,
                           help="Direct path to SCN data file.")
    data_args.add_argument("--filter_by_resolution",
                           type=my_bool,
                           default="True",
                           help="Whether or not to use dynamic batching when training.")
    data_args.add_argument("--complete_structures_only",
                           type=my_bool,
                           default="False",
                           help="If True, skip structures missing any residues.")
    data_args.add_argument("--dynamic_batching",
                           type=my_bool,
                           default="False",
                           help="Whether or not to use dynamic batching when training.")
    data_args.add_argument("--scn_dir",
                           type=str,
                           default="/home/jok120/sidechainnet_data",
                           help="Path to directory holding SidechainNet data files.")
    data_args.add_argument("--num_workers",
                           type=int,
                           help="Number of workers for each DataLoader.",
                           default=mp.cpu_count() // 2)
    data_args.add_argument("--shuffle",
                           type=my_bool,
                           help="If True, shuffle dataloaders (default=True).",
                           default="True")

    # Model-specific args
    parser = LitSidechainTransformer.add_model_specific_args(parser)

    # General args that may apply to multiple models
    training = parser.add_argument_group("Shared")
    training.add_argument('--batch_size', "-b", type=int, default=8)
    training.add_argument('--dropout', type=float, default=0.1, help="Layer dropout.")

    # Optimization args
    training = parser.add_argument_group("Optimization")
    training.add_argument('--loss_name',
                          '-l',
                          choices=["mse", "mse_openmm", "openmm"],
                          default="mse",
                          help="Loss used to train the model. Can be root mean squared "
                          "error (RMSE), distance-based root mean squared distance "
                          "(DRMSD), length-normalized DRMSD (ln-DRMSD) or a combination "
                          "of RMSE and ln-DRMSD.")
    training.add_argument('--opt_name',
                          '-opt',
                          type=str,
                          choices=['adam', 'sgd', 'adamw'],
                          default='sgd',
                          help="Training optimizer.")
    training.add_argument("--opt_lr", "-lr", type=float, default=1e-4)
    training.add_argument('--opt_lr_scheduling',
                          type=str,
                          choices=['noam', 'plateau'],
                          default='plateau',
                          help="noam: Use LR as described in Transformer paper, plateau:"
                          " Decrease learning rate after Validation loss plateaus.")
    training.add_argument('--opt_patience',
                          type=int,
                          default=10,
                          help="Patience for LR or chkpt routines.")
    training.add_argument('--opt_min_delta',
                          type=float,
                          default=0.01,
                          help="Threshold for considering improvements during training/lr"
                          " scheduling.")
    training.add_argument("--opt_weight_decay",
                          type=float,
                          default=0.0,
                          help="Applies weight decay to model weights.")
    training.add_argument('--opt_n_warmup_steps',
                          '-nws',
                          type=int,
                          default=10_000,
                          help="Number of warmup train steps when using lr-scheduling.")
    training.add_argument('--opt_lr_scheduling_metric',
                          choices=['val_loss', 'val_acc'],
                          default='val_acc',
                          help="Metric to use for early stopping, chkpts, etc. Choose "
                          "validation loss or accuracy.")
    training.add_argument("--loss_combination_weight",
                          type=float,
                          default=0.5,
                          help="When combining losses, use weight w where "
                          "loss = w * LossA + (1-w) * LossB.")
    training.add_argument("--overfit_batches_small",
                          type=my_bool,
                          default="True",
                          help="If true, overfit the smallest batch. Else, start with "
                          "something larger.")
    training.add_argument("--early_stopping",
                          type=my_bool,
                          default="True",
                          help="If true, use early stopping callback.")


    # Callbacks
    callback_args = parser.add_argument_group("Callbacks")
    callback_args.add_argument("--use_swa", type=my_bool, default="False")
    callback_args.add_argument("--swa_epoch_start", type=float, default=None)
    callback_args.add_argument("--use_batch_gradient_verification",
                               type=my_bool,
                               default="False")
    callback_args.add_argument("--auto_lr_find_custom", type=my_bool, default="False")

    return parser


def init_wandb(use_cluster, project, entity, model, dict_args):
    """Initialize Weights&Biases logger and local file directories.

    Args:
        use_cluster (bool): If True, modify some options for cluster runs.
        project (str): Name of W&B project.
        entity (str): Entity of W&B project.
        name (str): Name of W&B run.
        model (pl.LightningModule): PyTorch Lightning module.
        dict_args (dict): Command line arguments as a dictionary.

    Returns:
        tuple: Returns the Pytorch Lightning W&B logger and the checkpoint directory.
    """
    wandb_dir = "/scr/jok120/wandb" if use_cluster else os.path.expanduser("~/scr")
    if wandb_dir:
        os.makedirs(wandb_dir, exist_ok=True)
    logger = WandbLogger(project=project,
                         entity=entity,
                         dir=wandb_dir,
                         save_code=True,
                         log_model=True)
    logger.experiment.config.update(dict_args, allow_val_change=True)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.experiment.config.update({
        "n_params": n_params,
        "n_trainable_params": n_trainable_params,
        "casp_version": dict_args["casp_version"],
        "casp_thinning": dict_args["casp_thinning"]
    })
    logger.watch(model, "all")

    # Create directories
    local_dir = os.path.join(wandb_dir, wandb.run.dir)
    checkpoint_dir = os.path.join(local_dir, "checkpoints")
    os.makedirs(local_dir, exist_ok=True)
    os.makedirs(os.path.join(local_dir, "pdbs"), exist_ok=True)
    os.makedirs(os.path.join(local_dir, "pngs"), exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(local_dir, "MODEL.txt"), "w") as f:
        f.write(str(model) + "\n")
    model.save_dir = local_dir

    return logger, checkpoint_dir


def main():
    """Argument parsing, model loading, and model training."""
    # Parse args and update some defaults
    parser = create_parser()
    args = parser.parse_args()
    dict_args = vars(args)
    dict_args.update(
        deterministic=True,
        gpus=1 if args.gpus is None else args.gpus,
        gradient_clip_val=1 if args.gradient_clip_val is None else args.gradient_clip_val,
        gradient_clip_algorithm='value'
        if args.gradient_clip_algorithm is None else args.gradient_clip_algorithm,
        auto_select_gpus=True,
    )

    # Prepare torch
    pl.seed_everything(args.seed)

    # Load dataset
    data = scn.load(**dict_args, with_pytorch='dataloaders')
    data_module = LitSCNDataModule(data, batch_size=args.batch_size)

    # Update args with dataset information
    dict_args['angle_means'] = data_module.get_train_angle_means(6, None)
    dict_args['dataloader_name_mapping'] = data_module.val_dataloader_idx_to_name
    if dict_args['opt_lr_scheduling_metric'] == 'val_acc':
        tgt = 'acc'
        target_monitor_loss = f'metrics/valid/{data_module.val_dataloader_target}_{tgt}'
    else:
        tgt = 'rmse' if args.loss_name == 'mse' else args.loss_name
        target_monitor_loss = f'losses/valid/{data_module.val_dataloader_target}_{tgt}'
    if dict_args['overfit_batches'] > 0:
        tgt = 'rmse' if args.loss_name == 'mse' else args.loss_name
        target_monitor_loss = f"losses/train/{tgt}_step"
    dict_args['opt_lr_scheduling_metric'] = target_monitor_loss

    # Prepare model
    if args.model == "scn-trans-enc":
        # First, correct the number of heads so that d_in is divisible by n_heads
        while dict_args["d_in"] % dict_args["n_heads"] != 0:
            dict_args["n_heads"] -= 1
        model = LitSidechainTransformer(**dict_args)
        model.make_example_input_array(data_module)  # For auto batch size determination
    else:
        raise argparse.ArgumentError(f"Model architecture {args.model} not implemented.")

    # Prepare Weights and Biases logging
    wandb_logger, checkpoint_dir = init_wandb(use_cluster=args.cluster,
                                              project="sidechain-transformer",
                                              entity="koes-group",
                                              model=model,
                                              dict_args=dict_args)
    dict_args.update(
        logger=wandb_logger,
        default_root_dir=model.save_dir,
    )
    print(args, "\n")

    # Make callbacks
    my_callbacks = []
    if args.early_stopping:
        my_callbacks.append(
            callbacks.EarlyStopping(
                monitor=target_monitor_loss,
                min_delta=args.opt_min_delta,
                patience=args.opt_patience,
                verbose=True,
                check_finite=True,
                mode='min' if 'acc' not in target_monitor_loss else 'max'))
    my_callbacks.append(callbacks.LearningRateMonitor(logging_interval='step'))
    if args.enable_checkpointing:
        my_callbacks.append(
            callbacks.ModelCheckpoint(dirpath=checkpoint_dir,
                                      filename='{epoch:03d}-{step}-{val_loss:.2f}',
                                      monitor=target_monitor_loss))
    # my_callbacks.append(ModuleDataMonitor(submodules=True))
    if args.use_batch_gradient_verification:
        my_callbacks.append(
            BatchGradientVerificationCallback(
                input_mapping=lambda x: tuple([y.requires_grad_(True) for y in x[0]])))
    if args.use_swa:
        my_callbacks.append(
            callbacks.StochasticWeightAveraging(swa_epoch_start=args.swa_epoch_start))

    # Create a trainer
    trainer = pl.Trainer.from_argparse_args(argparse.Namespace(**dict_args),
                                            callbacks=my_callbacks)

    # Look for largest batch size and optimal LR
    tuner = Tuner(trainer)
    data_module.set_train_dataloader_descending()
    if dict_args['auto_scale_batch_size']:
        tuner.scale_batch_size(model, datamodule=data_module, mode='power', init_val=1)
        wandb.config.update({"batch_size": data_module.batch_size}, allow_val_change=True)
    if dict_args["auto_lr_find_custom"]:
        lr_finder = tuner.lr_find(model, data_module)
        fig = lr_finder.plot(suggest=True)
        fig.savefig(os.path.join(model.save_dir, 'lr_finder.png'))
        wandb.save(os.path.join(model.save_dir, 'lr_finder.png'))
        model.opt_lr = model.hparams.opt_lr = lr_finder.suggestion()
        print('New learning rate:', model.opt_lr)
        wandb.config.update({"opt_lr": model.opt_lr}, allow_val_change=True)

    # Train the model
    trainer.fit(model, data_module)
    trainer.test(model, data_module)


if __name__ == '__main__':
    main()
