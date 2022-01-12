"""
Training models to predict sidechain conformations given backbone conformations.
    Author: Jonathan King
    Date: 01/12/2022
"""
CLUSTER = True
from tqdm import tqdm
if CLUSTER:
    from functools import partialmethod
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

import argparse
import random
import os

import numpy as np
import torch.optim as optim
import torch.utils.data

import wandb
import torch

from sidechainnet.examples.sidechain_only_models import SidechainTransformer
import sidechainnet as scn
from sidechainnet.examples.losses import mse_over_angles
from sidechainnet.examples.optim import ScheduledOptim


def train_epoch(model, data, optimizer, device):
    """One complete training epoch."""
    model.train()
    pbar = tqdm(data['train'], leave=False, unit="batch",
                dynamic_ncols=True) if not CLUSTER else data['train']

    for step, p in enumerate(pbar):
        optimizer.zero_grad()
        # Select out backbone and sidechaine angles
        bb_angs = p.angs[:, :, :6]
        sc_angs_true = p.angs[:, :, 6:]
        sc_angs_true = scn.structure.trig_transform(sc_angs_true).reshape(
            sc_angs_true.shape[0], sc_angs_true.shape[1], 12)  # (B x L x 12)

        # Stack model inputs into a single tensor
        model_in = torch.cat([bb_angs, p.secs, p.evos], dim=-1)

        # Move inputs to device
        model_in = model_in.to(device)
        int_seqs = p.int_seqs.to(device)
        sc_angs_true = sc_angs_true.to(device)

        # Predict sidechain angles given input and sequence
        sc_angs_pred = model(model_in, int_seqs)  # ( B x L x 12)
        # print(torch.isnan(sc_angs_pred).any(), torch.isnan(sc_angs_true).any())

        # Compute loss and step
        loss = mse_over_angles(sc_angs_true, sc_angs_pred)
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 1)
        optimizer.step()

        # Record performance metrics
        cpu_loss = loss.item()
        wandb.log({"Train Batch RMSE": np.sqrt(cpu_loss)})
        if not CLUSTER:
            pbar.set_description(
                '\r  - (Train) rmse={rmse:.4f}'.format(rmse=np.sqrt(cpu_loss)))


def eval_epoch(model, data, device, test_set=False):
    """
    One complete evaluation epoch.
    """
    losses = {}
    model.eval()
    data_splits = ['test'] if test_set else ['train', 'valid-10', 'valid-50', 'valid-90']
    for data_split in data_splits:
        batch_iter = tqdm(data[data_split], leave=False, unit="batch",
                          dynamic_ncols=True) if not CLUSTER else data[data_split]
        with torch.no_grad():
            total_loss = 0
            for step, p in enumerate(batch_iter):
                # Select out backbone and sidechaine angles
                bb_angs = p.angs[:, :, :6]
                sc_angs_true = p.angs[:, :, 6:]
                sc_angs_true = scn.structure.trig_transform(sc_angs_true).reshape(
                    sc_angs_true.shape[0], sc_angs_true.shape[1], 12)  # (B x L x 12)

                # Stack model inputs into a single tensor
                model_in = torch.cat([bb_angs, p.secs, p.evos], dim=-1)

                # Move inputs to device
                model_in = model_in.to(device)
                int_seqs = p.int_seqs.to(device)
                sc_angs_true = sc_angs_true.to(device)

                # Predict sidechain angles given input and sequence
                sc_angs_pred = model(model_in, int_seqs)

                # Reshape prediction to match true sidechain angles
                loss = mse_over_angles(sc_angs_true, sc_angs_pred)
                total_loss += loss.item()

            # Record performance metrics
            avg_loss = np.sqrt(total_loss / (step + 1))
            wandb.log({f"{data_split.capitalize()} Epoch RMSE": avg_loss})
            losses[data_split] = avg_loss
    return losses


def train_loop(model, data, optimizer, device, args, scheduler):
    """Model training control loop."""
    for epoch_i in range(args.epochs):
        print(f'[ Epoch {epoch_i} ]')
        train_epoch(model, data, optimizer, device)
        losses = eval_epoch(model, data, device)

        # Update LR
        if scheduler:
            scheduler.step(losses['valid-50'])
    test_loss = eval_epoch(model, data, device, test_set=True)
    wandb.log({"Test Loss": test_loss})


def make_model(args, angle_means):
    """Return requested model architecture. Currently only scn-trans-enc is supported."""
    # Angle means must be unmodified before this step.
    angle_means = torch.tensor(angle_means).view(1, 1, 6)
    angle_means = scn.structure.trig_transform(angle_means).view(1, 1, 12)

    if args.model == "scn-trans-enc":
        model = SidechainTransformer(d_seq_embedding=args.d_seq_embedding,
                                     d_nonseq_data=args.d_nonseq_data,
                                     d_feedforward=args.d_feedforward,
                                     d_in=args.d_in,
                                     d_out=args.d_out,
                                     n_heads=args.n_heads,
                                     n_layers=args.n_layers,
                                     dropout=args.dropout,
                                     activation='relu',
                                     batch_first=True,
                                     device=args.device,
                                     angle_means=angle_means,
                                     embed_sequence=args.embed_sequence)
    else:
        raise argparse.ArgumentError("Model architecture not implemented.")
    return model


def seed_rngs(args):
    """Seed all necessary random number generators."""
    torch.set_num_threads(1)  # Suggested for issues with deadlocks, etc.
    seed = args.seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    if torch.backends.cudnn.deterministic:
        print("[Info] cudnn.deterministic set to True. CUDNN-optimized code may be slow.")


def setup_model_optimizer_scheduler(args, device, angle_means):
    model = make_model(args, angle_means).to(device)

    # Prepare optimizer
    wd = 10e-3 if args.weight_decay else 0
    if args.optimizer == "adam":
        optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                               lr=args.learning_rate,
                               weight_decay=wd)
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(filter(lambda x: x.requires_grad, model.parameters()),
                              lr=args.learning_rate,
                              weight_decay=wd)

    # Prepare scheduler
    if args.lr_scheduling == "noam":
        optimizer = ScheduledOptim(optimizer, args.d_in, args.n_warmup_steps)
        scheduler = None
    else:
        # Construct an LR scheduler with patience = 5 and a factor of 1/10
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=args.patience,
            verbose=True,
            threshold=args.early_stopping_threshold)
    return model, optimizer, scheduler


def create_parser():

    def my_bool(s):
        return s != 'False'

    parser = argparse.ArgumentParser()

    # Required args
    required = parser.add_argument_group("Required Args")
    required.add_argument("--name", type=str, help="The model name.", default=None)

    # Training parameters
    training = parser.add_argument_group("Training Args")
    training.add_argument("-lr", "--learning_rate", type=float, default=1e-4)
    training.add_argument('-e', '--epochs', type=int, default=10)
    training.add_argument("-b", '--batch_size', type=int, default=8)
    training.add_argument('-es',
                          '--early_stopping',
                          type=int,
                          default=20,
                          help="Stops if training hasn't improved in X epochs")
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
        choices=["mse", "drmsd", "lndrmsd", "combined"],
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
        default=10,
        help="Number of epochs to wait before reducing LR for plateau scheduler.")
    training.add_argument(
        '--early_stopping_threshold',
        type=float,
        default=0.001,
        help="Threshold for considering improvements during training/lr scheduling.")
    training.add_argument('-opt',
                          '--optimizer',
                          type=str,
                          choices=['adam', 'sgd'],
                          default='sgd',
                          help="Training optimizer.")
    training.add_argument("-s",
                          "--seed",
                          type=int,
                          default=11_731,
                          help="The random number generator seed for numpy "
                          "and torch.")
    training.add_argument(
        "--combined_drmsd_weight",
        type=float,
        default=0.5,
        help="When combining losses, use weight w for loss = w * drmsd + (1-w) * mse.")
    training.add_argument("--batching_order",
                          type=str,
                          choices=["descending", "ascending", "binned-random"],
                          default="binned-random",
                          help="Method for constructuing minibatches of proteins w.r.t. "
                          "sequence length. Batches can be provided in descending/"
                          "ascending order, or 'binned-random' which keeps the sequences"
                          "in a batch similar, while randomizing the bins/batches.")
    training.add_argument('-d', '--device', type=str, default='cuda')

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
                            type=my_bool,
                            default="True",
                            help="Applies weight decay to model weights.")
    model_args.add_argument("--embed_sequence",
                            type=my_bool,
                            default="True",
                            help="Whether or not to use embedding "
                            "layer in the transformer model.")

    return parser


def main():
    """Argument parsing, model loading, and model training."""
    torch.set_printoptions(precision=5, sci_mode=False)

    # Parse args
    parser = create_parser()
    args = parser.parse_args()

    # Prepare torch
    seed_rngs(args)

    # Load dataset
    data = scn.load(12,
                    100,
                    with_pytorch='dataloaders',
                    batch_size=args.batch_size,
                    dynamic_batching=False,
                    filter_by_resolution=True,
                    complete_structures_only=True)
    angle_means = data['train'].dataset.angle_means[6:]

    # Prepare model
    device = torch.device('cuda')
    model, optimizer, scheduler = setup_model_optimizer_scheduler(
        args, device, angle_means)
    # Since n_heads may have changed, we update the arg value
    args.n_heads = model.n_heads

    # Prepare Weights and Biases logging
    wandb_dir = "/scr/jok120/wandb"
    if wandb_dir:
        os.makedirs(wandb_dir, exist_ok=True)
    wandb.init(project="sidechain-transformer", entity="koes-group", dir=wandb_dir)
    wandb.watch(model, "all")
    if not args.name:
        args.name = wandb.run.id
    wandb.config.update(args, allow_val_change=True)
    wandb.config.update({"data_creation_date": data['train'].dataset.created_on})
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb.config.update({
        "n_params": n_params,
        "n_trainable_params": n_trainable_params,
        "casp_version": data['train'].dataset.casp_version,
        "casp_thinning": data['train'].dataset.thinning
    })

    local_base_dir = wandb.run.dir
    with open(os.path.join(local_base_dir, "MODEL.txt"), "w") as f:
        f.write(str(model) + "\n")

    print(args, "\n")

    # Start training
    train_loop(model, data, optimizer, device, args, scheduler)


if __name__ == '__main__':
    main()
