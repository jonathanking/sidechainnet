import argparse

import torch

import sidechainnet
from sidechainnet.examples.rgn import RGN


def main():
    """ An example of model training. """
    data = sidechainnet.load_datasets("../../data/sidechainnet/sidechainnet_casp12_100.pkl")
    model = RGN(args.d_in, args.d_hidden, args.d_out, args.n_layers)



def train(data, model):



def train_epoch(model, training_data, validation_datasets, optimizer, device, args, log_writer, metrics, pool=None):
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
        metrics = do_train_batch_logging(metrics, losses, src_seq, optimizer, args, log_writer, batch_iter, START_TIME,
                                         pred, tgt_crds, step, validation_datasets, model, device)

    metrics = update_metrics_end_of_epoch(metrics, "train")

    return metrics



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Required args
    required = parser.add_argument_group("Required Args")
    required.add_argument('--data', help="Path to SidechainNet.", default="../data/proteinnet/casp12_200123_30.pkl")

    # Training parameters
    training = parser.add_argument_group("Training Args")
    training.add_argument("-lr", "--learning_rate", type=float, default=1e-4)
    training.add_argument('-e', '--epochs', type=int, default=10)
    training.add_argument("-b", '--batch_size', type=int, default=8)
    training.add_argument('-cg', '--clip', type=float, default=1, help="Gradient clipping value.")
    training.add_argument('--sequential_drmsd_loss', action="store_true",
                          help="Compute DRMSD loss without batch-level parallelization.")
    training.add_argument("--train_eval_downsample", type=float, default=0.10, help="Fraction of training set to "
                                                                                    "evaluate on each epoch.")

    # Model parameters
    model_args = parser.add_argument_group("Model Args")
    model_args.add_argument('-dm', '--d_model', type=int, default=512,
                            help="Dimension of each sequence item in the model. Each layer uses the same dimension for "
                                 "simplicity.")
    model_args.add_argument('-dih', '--d_inner_hid', type=int, default=2048,
                            help="Dimension of the inner layer of the feed-forward layer at the end of every "
                                 "Transformer"
                                 " block.")
    model_args.add_argument('-nl', '--n_layers', type=int, default=6,
                            help="Number of layers in the model. If using encoder/decoder model, the encoder and "
                                 "decoder"
                                 " both have this number of layers.")


    # Saving args
    saving_args = parser.add_argument_group("Saving Args")
    saving_args.add_argument('--log_structure_step', type=int, default=10,
                             help="Frequency of logging structure data during training.")
    saving_args.add_argument('--log_val_struct_step', '-lvs', type=int, default=50,
                             help="During training, make predictions on 1 structure from every validation set.")
    saving_args.add_argument('--log_wandb_step', type=int, default=1,
                             help="Frequency of logging to wandb during training.")
    saving_args.add_argument("--save_pngs", "-png", type=my_bool, default="True",
                             help="Save images when making structures.")
    saving_args.add_argument('--no_cuda', action='store_true')
    saving_args.add_argument('--restart', action='store_true', help="Does not resume training.")
    saving_args.add_argument('--load_chkpt', type=str, default=None,
                             help="Path from which to load a model checkpoint.")
    main()


