import argparse
import os

import pytorch_lightning as pl
import pytorch_lightning.callbacks as callbacks
import sidechainnet as scn
import torch
import torch.utils.data
from pl_bolts.callbacks import (BatchGradientVerificationCallback,
                                ModuleDataMonitor)
from pytorch_lightning.loggers import WandbLogger
from sidechainnet.examples.LitSidechainTransformer import (
    LitSCNDataModule, LitSidechainTransformer)

import wandb

# TODO: variable sq length https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html


def create_parser():
    """Create an argument parser for the program."""

    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    def my_bool(s):
        """Allows bools instead of boolean flags."""
        return s != 'False'

    parser = argparse.ArgumentParser()

    # Required args
    required = parser.add_argument_group("Required Args")
    required.add_argument("--name", type=str, help="The model name.", default=None)

    # Data arguments
    data_args = parser.add_argument_group("Data Args")
    data_args.add_argument("--casp_version",
                           type=int,
                           help="CASP Version for SidechainNet {7-12, 'debug'}.",
                           default=12)
    data_args.add_argument(
        "--casp_thinning",
        type=str,
        help="CASP thinning for SidechainNet {30,50,70,90,100, 'debug'}.",
        default=30)
    data_args.add_argument("--scn_data_file",
                           type=str,
                           help="Direct path to SCN data file.")

    # Training parameters
    training = parser.add_argument_group("Training Args")
    training.add_argument("-lr", "--learning_rate", type=float, default=1e-4)
    training.add_argument('-e', '--epochs', type=int, default=10)
    training.add_argument("-b", '--batch_size', type=int, default=8)
    training.add_argument(
        '-nws',
        '--n_warmup_steps',
        type=int,
        default=10_000,
        help="Number of warmup training steps when using lr-scheduling as proposed in the"
        "original Transformer paper.")
    training.add_argument('-cg',
                          '--clip',
                          type=float,
                          default=1,
                          help="Gradient clipping value.")
    training.add_argument(
        '-l',
        '--loss',
        choices=["mse", "mse_openmm", "openmm"],
        default="mse",
        help="Loss used to train the model. Can be root mean squared error (RMSE), "
        "distance-based root mean squared distance (DRMSD), length-normalized DRMSD "
        "(ln-DRMSD) or a combination of RMSE and ln-DRMSD.")
    training.add_argument(
        '--lr_scheduling',
        type=str,
        choices=['noam', 'plateau'],
        default='plateau',
        help='noam: Use learning rate scheduling as described in Transformer paper, '
        'plateau: Decrease '
        'learning rate after Validation loss plateaus.')
    training.add_argument(
        '--patience',
        type=int,
        default=5,
        help="Number of epochs to wait before reducing LR for plateau scheduler.")
    training.add_argument(
        '--early_stopping_threshold',
        type=float,
        default=0.01,
        help="Threshold for considering improvements during training/lr scheduling.")
    training.add_argument('-opt',
                          '--optimizer',
                          type=str,
                          choices=['adam', 'sgd', 'adamw'],
                          default='sgd',
                          help="Training optimizer.")
    training.add_argument("-s",
                          "--seed",
                          type=int,
                          default=11_731,
                          help="The random number generator seed for numpy "
                          "and torch.")
    training.add_argument(
        "--combined_loss_weight",
        type=float,
        default=0.5,
        help="When combining losses, use weight w for loss = w * LossA + (1-w) * LossB.")
    training.add_argument('-d', '--device', type=str, default='cuda')
    training.add_argument("--dynamic_batching",
                          type=my_bool,
                          default="False",
                          help="Whether or not to use dynamic batching when training.")
    training.add_argument("--filter_resolution",
                          type=my_bool,
                          default="True",
                          help="Whether or not to use dynamic batching when training.")
    training.add_argument("--complete_structures_only",
                          type=my_bool,
                          default="False",
                          help="If True, skip structures missing any residues.")
    training.add_argument("--stochastic_weight_averaging", type=str, default="False")
    training.add_argument("--accumulate_grad_batches",
                          type=int,
                          default=1,
                          help="How many batches to run before backwards.")

    # Model parameters
    model_args = parser.add_argument_group("Model Args")
    model_args.add_argument(
        '-m',
        '--model',
        type=str,
        default="scn-trans-enc",
        help="Model architecture name. Currenlt only supports 'scn-trans-enc'.")
    model_args.add_argument('-dse',
                            '--d_seq_embedding',
                            type=int,
                            default=512,
                            help="Dimension of sequence embedding.")
    model_args.add_argument('-dnsd',
                            '--d_nonseq_data',
                            type=int,
                            default=35,
                            help="Dimension of non-sequence input embedding.")
    model_args.add_argument('-do',
                            '--d_out',
                            type=int,
                            default=12,
                            help="Dimension of desired model output.")
    model_args.add_argument('-di',
                            '--d_in',
                            type=int,
                            default=256,
                            help="Dimension of desired transformer model input.")
    model_args.add_argument(
        '-dff',
        '--d_feedforward',
        type=int,
        default=2048,
        help="Dimmension of the inner layer of the feed-forward layer at the end of every"
        " Transformer block.")
    model_args.add_argument('-nh',
                            '--n_heads',
                            type=int,
                            default=8,
                            help="Number of attention heads.")
    model_args.add_argument(
        '-nl',
        '--n_layers',
        type=int,
        default=6,
        help="Number of layers in the model. If using encoder/decoder model, the encoder "
        "and decoder both have this number of layers.")
    model_args.add_argument('--dropout',
                            type=float,
                            default=0.1,
                            help="Dropout applied between layers.")
    model_args.add_argument("--weight_decay",
                            type=float,
                            default=0.0,
                            help="Applies weight decay to model weights.")
    model_args.add_argument("--embed_sequence",
                            type=my_bool,
                            default="True",
                            help="Whether or not to use embedding "
                            "layer in the transformer model.")
    model_args.add_argument("--transformer_activation",
                            type=str,
                            default="relu",
                            help="Activation for Transformer layers.")

    # Saving args
    saving_args = parser.add_argument_group("Saving Args")
    saving_args.add_argument('-c',
                             '--cluster',
                             type=my_bool,
                             default="False",
                             help="Set of parameters to facilitate training on a remote" +
                             " cluster. Limited I/O, etc.")

    return parser


def init_wandb(use_cluster, project, entity, name, model, args, data_module):
    wandb_dir = "/scr/jok120/wandb" if use_cluster else os.path.expanduser("~/scr")
    if wandb_dir:
        os.makedirs(wandb_dir, exist_ok=True)
    logger = WandbLogger(project=project,
                         entity=entity,
                         dir=wandb_dir,
                         save_code=True,
                         name=name,
                         log_model=True)
    logger.experiment.config.update(args, allow_val_change=True)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.experiment.config.update({
        "n_params": n_params,
        "n_trainable_params": n_trainable_params,
        "casp_version": args.casp_version,
        "casp_thinning": args.casp_thinning
    })
    logger.watch(model, "all")
    return logger, wandb_dir


def main():
    """Argument parsing, model loading, and model training."""
    torch.set_printoptions(precision=5, sci_mode=False)

    # Parse args
    parser = create_parser()
    args = parser.parse_args()

    # Prepare torch
    pl.seed_everything(args.seed)

    # Load dataset
    data = scn.load(
        # "debug",
        casp_version=args.casp_version,
        thinning=args.casp_thinning,
        with_pytorch='dataloaders',
        batch_size=args.batch_size,
        dynamic_batching=args.dynamic_batching,
        filter_by_resolution=args.filter_resolution,
        complete_structures_only=args.complete_structures_only,
        scn_dir="/home/jok120/sidechainnet_data",
        num_workers=16)
    data_module = LitSCNDataModule(data)

    # TODO : setup angle means in data module/model
    angle_means = data['train'].dataset.angle_means[6:]
    angle_means = torch.tensor(angle_means).view(1, 1, 6)
    angle_means = scn.structure.trig_transform(angle_means).view(1, 1, 12)

    target = 'rmse' if args.loss == 'mse' else args.loss
    target_monitor_loss = f'valid/{data_module.val_dataloader_target}_{target}'

    # Prepare model
    if args.model == "scn-trans-enc":
        model = LitSidechainTransformer(
            # Model arguments
            d_seq_embedding=args.d_seq_embedding,
            d_feedforward=args.d_feedforward,
            d_nonseq_data=args.d_nonseq_data,
            d_in=args.d_in,
            d_out=args.d_out,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            dropout=args.dropout,
            activation=args.transformer_activation,
            angle_means=angle_means,
            embed_sequence=args.embed_sequence,
            # Optimizer and LR Scheduling
            loss_name=args.loss,
            opt_name=args.optimizer,
            opt_lr=args.learning_rate,
            opt_lr_scheduling=args.lr_scheduling,
            opt_lr_scheduling_metric=target_monitor_loss,
            opt_patience=args.patience,
            opt_early_stopping_threshold=args.early_stopping_threshold,
            opt_weight_decay=args.weight_decay,
            opt_n_warmup_steps=args.n_warmup_steps,
            dataloader_name_mapping=data_module.val_dataloader_idx_to_name)
        # Since n_heads may have changed, we update the arg value
        args.n_heads = model.n_heads
        example_batch = data_module.get_example_input_array()
        non_seq_data = model._prepare_model_input(example_batch)[0]
        model.example_input_array = non_seq_data, example_batch.seqs_int
    else:
        raise argparse.ArgumentError(f"Model architecture {args.model} not implemented.")

    # Prepare Weights and Biases logging
    wandb_logger, wandb_dir = init_wandb(use_cluster=args.cluster,
                                         project="sidechain-transformer",
                                         entity="koes-group",
                                         name=args.name,
                                         model=model,
                                         args=args,
                                         data_module=data_module)

    global LOCAL_DIR
    LOCAL_DIR = os.path.join(wandb_dir, wandb.run.dir)
    checkpoint_dir = os.path.join(LOCAL_DIR, "checkpoints")
    os.makedirs(LOCAL_DIR, exist_ok=True)
    os.makedirs(os.path.join(LOCAL_DIR, "pdbs"), exist_ok=True)
    os.makedirs(os.path.join(LOCAL_DIR, "pngs"), exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(LOCAL_DIR, "MODEL.txt"), "w") as f:
        f.write(str(model) + "\n")
    model.save_dir = LOCAL_DIR

    print(args, "\n")

    # Make callbacks
    my_callbacks = []
    my_callbacks.append(
        callbacks.EarlyStopping(monitor=target_monitor_loss,
                                min_delta=0.01,
                                patience=args.patience,
                                verbose=True,
                                check_finite=True,
                                mode='min'))
    my_callbacks.append(callbacks.LearningRateMonitor(logging_interval='step'))
    my_callbacks.append(
        callbacks.ModelCheckpoint(dirpath=checkpoint_dir,
                                  filename='{epoch:03d}-{step}-{val_loss:.2f}',
                                  monitor=target_monitor_loss))
    # my_callbacks.append(ModuleDataMonitor(submodules=True))
    # my_callbacks.append(
    #     BatchGradientVerificationCallback(
    #         input_mapping=lambda x: tuple([y.requires_grad_(True) for y in x[0]])))
    # if args.stochastic_weight_averaging == "True":
    #     # Default epoch start = 0.8 * max_epochs
    #     my_callbacks.append(callbacks.StochasticWeightAveraging())
    # elif args.stochastic_weight_averaging != "False":
    #     my_callbacks.append(
    #         callbacks.StochasticWeightAveraging(
    #             swa_epoch_start=float(args.stochastic_weight_averaging)))

    # Begin Training
    trainer = pl.Trainer(
        default_root_dir=LOCAL_DIR,
        deterministic=True,
        accumulate_grad_batches=args.accumulate_grad_batches,  # Default 1
        # https://github.com/PyTorchLightning/pytorch-lightning/discussions/12073
        # auto_scale_batch_size="binsearch",
        gradient_clip_val=args.clip,
        gradient_clip_algorithm="value",
        callbacks=my_callbacks,
        logger=wandb_logger,
        limit_train_batches=300,
        gpus=1)

    # trainer.tune(model)  # Look for largest batch size

    trainer.fit(model, data_module)
    # trainer.fit(model, train_dataloader=data['train'], val_dataloaders=data['valid-10'])


if __name__ == '__main__':
    main()
