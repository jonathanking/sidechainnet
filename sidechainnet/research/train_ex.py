"""
Primary script for training models to predict protein structure from amino
acid sequence.
    Author: Jonathan King
    Date: 10/25/2019
"""

import argparse
import csv
import random

import torch.multiprocessing as mp
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm
import wandb

from protein_transformer.dataset import prepare_dataloaders, MAX_SEQ_LEN
from protein_transformer.log import *
from protein_transformer.losses import compute_batch_drmsd, mse_over_angles, combine_drmsd_mse
from protein_transformer.models.convolutional_encoder import ConvEncoderOnlyTransformer
from protein_transformer.models.encoder_only import EncoderOnlyTransformer
from protein_transformer.models.transformer.Optimizer import ScheduledOptim
from protein_transformer.models.transformer.Transformer import Transformer
from protein_transformer.protein.Structure import NUM_PREDICTED_ANGLES


def train_epoch(model,
                training_data,
                validation_datasets,
                optimizer,
                device,
                args,
                log_writer,
                metrics,
                pool=None):
    """
    One complete training epoch.
    """
    model.train()
    metrics = reset_metrics_for_epoch(metrics, "train")
    batch_iter = tqdm(training_data, leave=False, unit="batch", dynamic_ncols=True)
    for step, batch in enumerate(batch_iter):
        optimizer.zero_grad()
        src_seq, tgt_ang, tgt_crds = map(lambda x: x.to(device), batch)
        pred = model(src_seq, tgt_ang)
        losses = get_losses(args, pred, tgt_ang, tgt_crds, src_seq, pool=pool)

        # Clip gradients
        if args.clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        # Update parameters
        optimizer.step()

        # Record performance metrics
        metrics = do_train_batch_logging(metrics, losses, src_seq, optimizer, args,
                                         log_writer, batch_iter, START_TIME, pred,
                                         tgt_crds, step, validation_datasets, model,
                                         device)

    metrics = update_metrics_end_of_epoch(metrics, "train")

    return metrics


def get_losses(args,
               pred,
               tgt_ang,
               tgt_crds,
               src_seq,
               pool=None,
               log=True,
               do_backwards=True,
               return_rmsd=False,
               eval_mode=False):
    """
    Returns the computed losses/metrics for a batch. The variable 'loss'
    will differ depending on the loss the user requested to train on.
    """
    # TODO remove outdated reference to loss
    # Always compute MSE loss b/c it's computationally cheap.
    m_loss_full = mse_over_angles(pred, tgt_ang)
    m_loss_bb = mse_over_angles(pred, tgt_ang, bb_only=True)
    m_loss_sc = mse_over_angles(pred, tgt_ang, sc_only=True)

    if args.loss in ["lndrmsd", "drmsd", "combined"] or eval_mode:
        ls = compute_batch_drmsd(pred,
                                 tgt_crds,
                                 src_seq,
                                 do_backward=do_backwards,
                                 retain_graph=args.loss == "combined",
                                 pool=pool,
                                 backbone_only=args.backbone_loss,
                                 return_rmsd=return_rmsd)
        if return_rmsd:
            d_loss, ln_d_loss, d_bb_loss, d_bb_ln_loss, rmsd_loss = ls
        else:
            d_loss, ln_d_loss, d_bb_loss, d_bb_ln_loss, rmsd_loss = *ls, None
        c_loss = combine_drmsd_mse(ln_d_loss,
                                   m_loss_full,
                                   w=args.combined_drmsd_weight,
                                   log=log)
        if args.loss == "lndrmsd":
            loss = ln_d_loss
        elif args.loss == "drmsd":
            loss = d_loss
        elif args.loss == "combined":
            loss = c_loss
            if do_backwards:
                c_loss.backward()
        elif args.loss == "mse":
            loss = m_loss_full

    elif args.loss == "mse":
        # Other losses are not computed for computational efficiency.
        d_loss, ln_d_loss, d_bb_loss, d_bb_ln_loss, c_loss, rmsd_loss = torch.tensor(0), torch.tensor(0), \
                                                                        torch.tensor(0), torch.tensor(0), \
                                                                        torch.tensor(0), None
        loss = m_loss_full
        if do_backwards:
            m_loss_full.backward()

    losses = {
        "loss": loss,
        "drmsd-full": d_loss,
        "lndrmsd-full": ln_d_loss,
        "drmsd-bb": d_bb_loss,
        "lndrmsd-bb": d_bb_ln_loss,
        "combined-full": c_loss,
        "mse-full": m_loss_full,
        "mse-bb": m_loss_bb,
        "mse-sc": m_loss_sc,
        "rmsd-full": rmsd_loss
    }

    return losses


def eval_epoch(model, validation_data, device, args, metrics, mode="valid", pool=None):
    """
    One compete evaluation epoch.
    """
    model.eval()
    metrics = reset_metrics_for_epoch(metrics, mode)
    batch_iter = tqdm(validation_data,
                      mininterval=.5,
                      leave=False,
                      unit="batch",
                      dynamic_ncols=True)

    with torch.no_grad():
        for batch in batch_iter:
            src_seq, tgt_ang, tgt_crds = map(lambda x: x.to(device), batch)
            pred = model(src_seq, tgt_ang)

            losses = get_losses(args,
                                pred,
                                tgt_ang,
                                tgt_crds,
                                src_seq,
                                pool=pool,
                                do_backwards=False,
                                eval_mode=True,
                                return_rmsd=True)

            # Record performance metrics
            metrics = do_eval_batch_logging(metrics, losses, src_seq, args, batch_iter,
                                            pred, tgt_crds, mode)

    do_eval_epoch_logging(metrics, mode)

    return metrics


def train(model, metrics, training_data, train_eval_loader, validation_datasets,
          test_data, optimizer, device, args, log_writer, scheduler, drmsd_worker_pool):
    """
    Model training control loop.
    """
    for epoch_i in range(START_EPOCH, args.epochs):
        print(f'[ Epoch {epoch_i} ]')

        # Train epoch
        start = time.time()
        metrics = train_epoch(model,
                              training_data,
                              validation_datasets,
                              optimizer,
                              device,
                              args,
                              log_writer,
                              metrics,
                              pool=drmsd_worker_pool)
        if args.eval_train:
            metrics = eval_epoch(model,
                                 train_eval_loader,
                                 device,
                                 args,
                                 metrics,
                                 mode="train",
                                 pool=drmsd_worker_pool)
        print_end_of_epoch_status("train", (start, metrics))
        log_batch(log_writer, metrics, START_TIME, mode="train", end_of_epoch=True)

        # Valid epoch
        if not args.train_only:
            for split, validation_data in validation_datasets.items():
                start = time.time()
                metrics = eval_epoch(model,
                                     validation_data,
                                     device,
                                     args,
                                     metrics,
                                     pool=drmsd_worker_pool,
                                     mode=f"valid-{split}")
                print_end_of_epoch_status(f"valid-{split}", (start, metrics))
                log_batch(log_writer,
                          metrics,
                          START_TIME,
                          mode=f"valid-{split}",
                          end_of_epoch=True)
            log_avg_validation_performance(metrics, validation_datasets)

        # Update LR
        if scheduler:
            scheduler.step(metrics[args.es_mode][f"epoch-{args.es_metric}-full"])

        # Checkpointing
        try:
            metrics = update_loss_trackers(args, epoch_i, metrics)
        except EarlyStoppingCondition:
            break
        checkpoint_model(args, optimizer, model, metrics, epoch_i, scheduler)

    # Test Epoch
    if not args.train_only:
        start = time.time()
        metrics = eval_epoch(model,
                             test_data,
                             device,
                             args,
                             metrics,
                             mode="test",
                             pool=drmsd_worker_pool)
        print_end_of_epoch_status("test", (start, metrics))
        log_batch(log_writer, metrics, START_TIME, mode="test", end_of_epoch=True)

    if drmsd_worker_pool:
        drmsd_worker_pool.close()
        drmsd_worker_pool.join()


def checkpoint_model(args, optimizer, model, metrics, epoch_i, scheduler):
    """
    Records model state according to a checkpointing policy. Defaults to best
    validation set performance. Returns True iff model was saved.
    """
    cur_loss, loss_history = metrics["loss_to_compare"], metrics["losses_to_compare"]
    if args.checkpoint_time_interval == 0:
        do_time_chkpt = False
    else:
        do_time_chkpt = (time.time() - metrics["last_chkpt_time"]
                        ) / 3600 > args.checkpoint_time_interval

    if len(loss_history) == 1 or cur_loss < min(loss_history[:-1]):
        modifier = "best"
    elif do_time_chkpt:
        modifier = "latest"
    else:
        return False

    chkpt_file_name = args.chkpt_path + f"_{modifier}.chkpt"
    wandb.run.summary[f"{modifier}_validation_loss"] = cur_loss
    wandb.run.summary[f"{modifier}_validation_epoch"] = epoch_i

    model_state_dict = model.state_dict()
    checkpoint = {
        'model_state_dict': model_state_dict,
        'settings': args,
        'epoch': epoch_i,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': cur_loss,
        'metrics': metrics,
        'elapsed_time': time.time() - START_TIME
    }

    torch.save(checkpoint, chkpt_file_name)
    wandb.save(chkpt_file_name)
    if not args.train_only:
        wandb.run.summary["avg_evaluation_speed"] = np.mean(
            metrics["valid-70"]["speed-history"])
    wandb.run.summary["avg_training_speed"] = np.mean(metrics["train"]["speed-history"])
    metrics["last_chkpt_time"] = time.time()
    print('\r    - [Info] The checkpoint file has been updated.')

    return True


def load_model(model, optimizer, scheduler, args):
    """
    Given a model, its optimizer, and the program's arguments, resumes model
    training if the user has not specified otherwise. Assumes model was saved
    the 'best' mode.
    """
    global START_EPOCH
    global START_TIME
    if args.load_chkpt:
        chkpt_file_name = args.load_chkpt
    else:
        chkpt_file_name = args.chkpt_path + "_best.chkpt"

    # Try to load the model checkpoint, if it exists
    if os.path.exists(chkpt_file_name) and not args.restart:
        print(f"[Info] Attempting to load model from {chkpt_file_name}.")
    else:
        return model, optimizer, scheduler, False, init_metrics(args)
    checkpoint = torch.load(chkpt_file_name)
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError as e:
        print("[Info] Error loading model.")
        print(e)
        exit(1)

    # Load the optimizer state by default
    if not args.restart_opt:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Reload the scheduler's state_dict
    if scheduler:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    START_EPOCH = checkpoint['epoch'] + 1
    START_TIME -= checkpoint['elapsed_time']
    print(
        f"[Info] Resuming model training from end of Epoch {checkpoint['epoch']}. Previous validation loss"
        f" = {checkpoint['loss']:.4f}.")
    return model, optimizer, scheduler, True, checkpoint['metrics']


def make_model(args, device, angle_means):
    """
    Returns requested model architecture. Currently only enc-only and enc-dec
    are supported.
    """
    if args.model == "enc-only":
        model = EncoderOnlyTransformer(nlayers=args.n_layers,
                                       nhead=args.n_head,
                                       dmodel=args.d_model,
                                       dff=args.d_inner_hid,
                                       max_seq_len=MAX_SEQ_LEN,
                                       dropout=args.dropout,
                                       vocab=VOCAB,
                                       angle_means=angle_means,
                                       use_tanh_out=not "linear-out" in args.model)
    elif "conv-enc" in args.model:
        model = ConvEncoderOnlyTransformer(
            nlayers=args.n_layers,
            nhead=args.n_head,
            dmodel=args.d_model,
            dff=args.d_inner_hid,
            max_seq_len=MAX_SEQ_LEN,
            dropout=args.dropout,
            vocab=VOCAB,
            angle_means=angle_means,
            use_tanh_out="linear-out" not in args.model,
            conv_kernel_sizes=[
                a for a in [args.conv1_size, args.conv2_size, args.conv3_size] if a
            ],
            conv_dim_reductions=[
                a for a in [args.conv1_reduc, args.conv2_reduc, args.conv3_reduc] if a
            ],
            use_embedding=args.use_embedding,
            conv_out_matches_dm=args.conv_out_matches_dm)
    elif args.model == "enc-dec":
        model = Transformer(dm=args.d_model,
                            dff=args.d_inner_hid,
                            din=len(VOCAB),
                            dout=NUM_PREDICTED_ANGLES * 2,
                            n_heads=args.n_head,
                            n_enc_layers=args.n_layers,
                            n_dec_layers=args.n_layers,
                            max_seq_len=MAX_SEQ_LEN,
                            pad_char=VOCAB.pad_id,
                            missing_coord_filler=MISSING_COORD_FILLER,
                            device=device,
                            dropout=args.dropout,
                            fraction_complete_tf=args.fraction_complete_tf,
                            fraction_subseq_tf=args.fraction_subseq_tf,
                            angle_means=angle_means)
    else:
        raise argparse.ArgumentError("Model architecture not implemented.")
    return model


def parse_conv_kernel_info_from_model_name(mname):
    """ Returns the parsed settings for the number and arrangement of convolutional
    layers. Specifically, this returns the requested kernel sizes and the factor
    by which the number of channels should be reduced through each layer.
    Ex:
    >>> parse_conv_kernel_info_from_model_name("conv-enc|3,7,11|2,2,2")
    ([3, 7, 11], [2, 2, 2])
    """
    try:
        _, kernel_sizes, dim_reducs = mname.split("|")
    except ValueError:
        return [], []
    kernel_sizes = list(map(int, kernel_sizes.split(",")))
    dim_reducs = list(map(float, dim_reducs.split(",")))
    return kernel_sizes, dim_reducs


def seed_rngs(args):
    """
    Seed all necessary random number generators.
    """
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


def init_worker_pool(args):
    """
    Creates the worker pool for drmsd batch computation. Does nothing if sequential.
    """
    torch.multiprocessing.set_start_method("spawn")
    return torch.multiprocessing.Pool(
        mp.cpu_count()) if not args.sequential_drmsd_loss else None


def setup_model_optimizer_scheduler(args, device, angle_means):
    model = make_model(args, device, angle_means).to(device)

    # Prepare optimizer
    wd = 10e-3 if args.weight_decay else 0
    if args.optimizer == "adam":
        optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                               betas=(0.9, 0.98),
                               eps=1e-09,
                               lr=args.learning_rate,
                               weight_decay=wd)
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(filter(lambda x: x.requires_grad, model.parameters()),
                              lr=args.learning_rate,
                              weight_decay=wd)

    # Prepare scheduler
    if args.lr_scheduling == "noam":
        optimizer = ScheduledOptim(optimizer, args.d_model, args.n_warmup_steps)
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
    required.add_argument('--data',
                          help="Path to training data.",
                          default="../data/proteinnet/casp12_200123_30.pt")
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
        help=
        "Number of warmup training steps when using lr-scheduling as proposed in the original"
        "Transformer paper.")
    training.add_argument('-cg',
                          '--clip',
                          type=float,
                          default=1,
                          help="Gradient clipping value.")
    training.add_argument(
        '-l',
        '--loss',
        choices=["mse", "drmsd", "lndrmsd", "combined"],
        default="combined",
        help=
        "Loss used to train the model. Can be root mean squared error (RMSE), distance-based "
        "root mean squared distance (DRMSD), length-normalized DRMSD (ln-DRMSD) or a combination"
        " of RMSE and ln-DRMSD.")
    training.add_argument(
        '--train_only',
        action='store_true',
        help=
        "Train, validation, and testing sets are the same. Only report train accuracy.")
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
    training.add_argument(
        '-esm',
        '--early_stopping_metric',
        choices=[
            f"{mode}-{metric}" for metric in ["mse", "drmsd", "lndrmsd", "combined"]
            for mode in ["train", "test"] + [f"valid-{split}" for split in VALID_SPLITS]
        ],
        default=None,
        help="Metric observed for early stopping and LR scheduling.")
    training.add_argument(
        '--without_angle_means',
        action='store_true',
        help="Do not initialize the model with pre-computed angle means.")
    training.add_argument(
        '--eval_train',
        type=my_bool,
        default="False",
        help="Perform an evaluation of the entire training set after a training epoch.")
    training.add_argument('-opt',
                          '--optimizer',
                          type=str,
                          choices=['adam', 'sgd'],
                          default='sgd',
                          help="Training optimizer.")
    training.add_argument(
        "-fctf",
        "--fraction_complete_tf",
        type=float,
        default=1,
        help=
        "Fraction of the time to use teacher forcing for every timestep of the batch. Model "
        "trains fastest when this is 1.")
    training.add_argument(
        "-fsstf",
        "--fraction_subseq_tf",
        type=float,
        default=1,
        help="Fraction of the time to use teacher forcing on a per-timestep basis.")
    training.add_argument(
        "--skip_missing_res_train",
        type=my_bool,
        default="False",
        help=
        "When training, skip over batches that have missing residues. This can make training"
        "faster if using teacher forcing.")
    training.add_argument(
        "--repeat_train",
        type=int,
        default=1,
        help="Duplicate the training set X times. Useful for training on small datasets.")
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
    training.add_argument('--backbone_loss',
                          action='store_true',
                          help="While training, only evaluate loss on the backbone.")
    training.add_argument('--sequential_drmsd_loss',
                          action="store_true",
                          help="Compute DRMSD loss without batch-level parallelization.")
    training.add_argument("--bins",
                          type=int,
                          default=-1,
                          help="Number of bins for protein dataset batching. ")
    training.add_argument("--train_eval_downsample",
                          type=float,
                          default=0.10,
                          help="Fraction of training set to "
                          "evaluate on each epoch.")
    training.add_argument("--automatically_determine_batch_size",
                          "-adbs",
                          type=my_bool,
                          help="Experimentally determine"
                          "the maximum allowable batch"
                          "size for training.",
                          default="False")

    # Model parameters
    model_args = parser.add_argument_group("Model Args")
    model_args.add_argument(
        '-m',
        '--model',
        type=str,
        default="enc-only",
        help="Model architecture type. Encoder only or encoder/decoder model.")
    model_args.add_argument(
        '-dm',
        '--d_model',
        type=int,
        default=512,
        help=
        "Dimension of each sequence item in the model. Each layer uses the same dimension for "
        "simplicity.")
    model_args.add_argument(
        '-dih',
        '--d_inner_hid',
        type=int,
        default=2048,
        help="Dimmension of the inner layer of the feed-forward layer at the end of every "
        "Transformer"
        " block.")
    model_args.add_argument('-nh',
                            '--n_head',
                            type=int,
                            default=8,
                            help="Number of attention heads.")
    model_args.add_argument(
        '-nl',
        '--n_layers',
        type=int,
        default=6,
        help=
        "Number of layers in the model. If using encoder/decoder model, the encoder and "
        "decoder"
        " both have this number of layers.")
    model_args.add_argument('-do',
                            '--dropout',
                            type=float,
                            default=0.1,
                            help="Dropout applied between layers.")
    model_args.add_argument(
        '--postnorm',
        action='store_true',
        help=
        "Use post-layer normalization, as depicted in the original figure for the Transformer "
        "model. May not train as well as pre-layer normalization.")
    model_args.add_argument("--weight_decay",
                            type=my_bool,
                            default="True",
                            help="Applies weight decay to model weights.")
    model_args.add_argument("--conv1_size",
                            type=int,
                            default=None,
                            help="Size of conv1 layer kernel for 'conv-enc' model.")
    model_args.add_argument("--conv2_size",
                            type=int,
                            default=None,
                            help="Size of conv2 layer kernel for 'conv-enc' model.")
    model_args.add_argument("--conv3_size",
                            type=int,
                            default=None,
                            help="Size of conv2 layer kernel for 'conv-enc' model.")
    model_args.add_argument(
        "--conv1_reduc",
        type=int,
        default=None,
        help=
        "Factor by which conv1 layer reduces the number of channels for 'conv-enc' model."
    )
    model_args.add_argument(
        "--conv2_reduc",
        type=int,
        default=None,
        help=
        "Factor by which conv2 layer reduces the number of channels for 'conv-enc' model."
    )
    model_args.add_argument(
        "--conv3_reduc",
        type=int,
        default=None,
        help=
        "Factor by which conv2 layer reduces the number of channels for 'conv-enc' model."
    )
    model_args.add_argument("--use_embedding",
                            type=my_bool,
                            default="True",
                            help="Whether or not to use embedding "
                            "layer in the transformer model.")
    model_args.add_argument(
        "--conv_out_matches_dm",
        type=my_bool,
        default="True",
        help="If True, the final convolution layer at the start of the model will match the"
        " dimensionality of the requested d_model. Used for ConvEnc models.")

    # Saving args
    saving_args = parser.add_argument_group("Saving Args")
    saving_args.add_argument('--log_structure_step',
                             type=int,
                             default=10,
                             help="Frequency of logging structure data during training.")
    saving_args.add_argument(
        '--log_val_struct_step',
        '-lvs',
        type=int,
        default=50,
        help="During training, make predictions on 1 structure from every validation set."
    )
    saving_args.add_argument('--log_wandb_step',
                             type=int,
                             default=1,
                             help="Frequency of logging to wandb during training.")
    saving_args.add_argument("--save_pngs",
                             "-png",
                             type=my_bool,
                             default="True",
                             help="Save images when making structures.")
    saving_args.add_argument('--no_cuda', action='store_true')
    saving_args.add_argument('-c',
                             '--cluster',
                             type=my_bool,
                             default="False",
                             help="Set of parameters to facilitate training on a remote" +
                             " cluster. Limited I/O, etc.")
    saving_args.add_argument('--restart',
                             action='store_true',
                             help="Does not resume training.")
    saving_args.add_argument(
        '--restart_opt',
        action='store_true',
        help="Resumes training but does not load the optimizer state. ")
    saving_args.add_argument(
        "--checkpoint_time_interval",
        type=float,
        default=0,
        help="The amount of time (in hours) after which a model checkpoint is made, "
        "regardless of its performance. ")
    saving_args.add_argument('--load_chkpt',
                             type=str,
                             default=None,
                             help="Path from which to load a model checkpoint.")
    return parser


def determine_largest_batch_size(args, fraction_to_keep=0.8):
    """
    Repeatedly tries a few training batches until the system runs out of memory. Returns the largest batch size
    found. Uses a completely separate script in order to avoid issues with CUDA memory not being cleared.
    """
    import subprocess
    from math import ceil
    print("Determining maximum batch size.")
    b = 1
    start = time.time()
    completed_process = subprocess.run(args=[
        "python", "../scripts/determine_largest_batchsize.py", *sys.argv[1:],
        "--experimental_batch_size",
        str(b)
    ],
                                       encoding="utf-8")
    b = int(completed_process.returncode)
    if args.loss != "mse" and b <= torch.multiprocessing.cpu_count():
        max_batch_size = b
    else:
        max_batch_size = max(1, ceil((b * fraction_to_keep)))
    print(
        f"Maximum batch size found to be {b}. Will proceed with {max_batch_size}. {int((time.time() - start)//60)}"
        f" min elapsed.")
    return max_batch_size


def main():
    """
    Argument parsing, model loading, and model training.
    """
    global LOGFILEHEADER
    global START_EPOCH
    global START_TIME
    global MISSING_COORD_FILLER

    # Fix file descriptor issue: https://github.com/pytorch/pytorch/issues/973
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

    START_EPOCH = 0
    START_TIME = time.time()
    MISSING_COORD_FILLER = 0  # Used when teacher forcing with an encoder/decoder model

    torch.set_printoptions(precision=5, sci_mode=False)

    # Parse args
    parser = create_parser()
    args = parser.parse_args()
    args.cuda = not args.no_cuda
    assert not args.name or "_" not in args.name, "Please do not use a '_' in your model name. " \
                                 "Conflicts with structure files."
    args.buffering_mode = 1
    if not args.early_stopping_metric:
        args.early_stopping_metric = f"train-{args.loss}"
    args.es_mode, args.es_metric = args.early_stopping_metric.split("-")
    args.add_sos_eos = args.model == "enc-dec"
    LOGFILEHEADER = prepare_log_header(args)
    args.bins = "auto" if args.bins == -1 else args.bins
    if args.automatically_determine_batch_size:
        args.batch_size = determine_largest_batch_size(args)
    if "conv-enc" in args.model:  # This will generate a model architecture based on a supplied name, ie conv-env|3,3,3|2,2,2
        kernel_sizes, dim_reducs = parse_conv_kernel_info_from_model_name(args.model)
        assert len(
            kernel_sizes) <= 3, "Only 3 convolution layers are currently supported."
        if len(kernel_sizes) >= 1:
            args.conv1_size = kernel_sizes[0]
            args.conv1_reduc = dim_reducs[0]
        if len(kernel_sizes) >= 2:
            args.conv2_size = kernel_sizes[1]
            args.conv2_reduc = dim_reducs[1]
        if len(kernel_sizes) == 3:
            args.conv3_size = kernel_sizes[2]
            args.conv3_reduc = dim_reducs[2]
        args.model = "conv-enc"

    # Prepare torch
    drmsd_worker_pool = init_worker_pool(args)
    seed_rngs(args)

    # Load dataset
    data = torch.load(args.data)
    args.max_token_seq_len = data['settings']["max_len"]
    angle_means = data["settings"]["angle_means"]

    # Prepare model
    device = torch.device('cuda' if args.cuda else 'cpu')
    model, optimizer, scheduler = setup_model_optimizer_scheduler(
        args, device, angle_means)

    # Prepare Weights and Biases logging
    wandb_dir = "/scr/jok120/wandb" if args.cluster and os.path.isdir("/scr") else None
    if wandb_dir:
        os.makedirs(wandb_dir, exist_ok=True)
    wandb.init(project="protein-transformer",
               entity="koes-group",
               dir=wandb_dir,
               name=args.name)
    wandb.watch(model, "all")
    if not args.name:
        args.name = wandb.run.id
    wandb.config.update(args, allow_val_change=True)
    if type(data["date"]) == set:
        wandb.config.update({"data_creation_date": next(iter(data["date"]))})
    else:
        wandb.config.update({"data_creation_date": data["date"]})
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb.config.update({
        "n_params": n_params,
        "n_trainable_params": n_trainable_params,
        "max_seq_len": MAX_SEQ_LEN
    })
    wandb.run.summary["stopped_training_early"] = False
    wandb.run.summary["max_batch_size"] = args.batch_size
    local_base_dir = wandb.run.dir
    args.structure_dir = f"../data/logs/structures/pdbs/{wandb.run.id}"
    args.gltf_dir = f"../data/logs/structures/gltfs/{wandb.run.id}"
    args.png_dir = f"../data/logs/structures/pngs/{wandb.run.id}"
    os.makedirs(args.structure_dir, exist_ok=True)
    os.makedirs(args.gltf_dir, exist_ok=True)
    os.makedirs(args.png_dir, exist_ok=True)
    with open(os.path.join(local_base_dir, "MODEL.txt"), "w") as f:
        f.write(str(model) + "\n")
    # Because some models use convolutional layers to change the dim of sequence elements prior to attention
    # layers, we will update wandb logging to account for the correct "model" dimension.
    wandb.config.update(
        {
            "d_model":
                model.encoder.conv_out_size() if "conv" in args.model else args.d_model,
            "d_model_start":
                args.d_model
        },
        allow_val_change=True)

    # Prepare log and checkpoint files
    args.chkpt_path = os.path.join(local_base_dir, "checkpoints")
    os.makedirs(args.chkpt_path, exist_ok=True)
    args.log_file = os.path.join(local_base_dir, args.name + '.train')
    print('[Info] Training performance will be written to file: {}'.format(args.log_file))
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    model, optimizer, scheduler, resumed, metrics = load_model(model, optimizer,
                                                               scheduler, args)
    if resumed:
        log_f = open(args.log_file, 'a', buffering=args.buffering_mode)
    else:
        log_f = open(args.log_file, 'w', buffering=args.buffering_mode)
        log_f.write(LOGFILEHEADER)
    log_writer = csv.writer(log_f)

    wandb.save(os.path.join(local_base_dir, "structures/*"))
    wandb.save(os.path.join(local_base_dir, "checkpoints/*"))
    wandb.save(os.path.join(local_base_dir, "*.train"))

    print(args, "\n")

    # Start training
    training_data, training_eval_loader, validation_datasets, test_data = prepare_dataloaders(
        data, args, MAX_SEQ_LEN)
    del data
    train(model, metrics, training_data, training_eval_loader, validation_datasets,
          test_data, optimizer, device, args, log_writer, scheduler, drmsd_worker_pool)
    log_f.close()


if __name__ == '__main__':
    main()