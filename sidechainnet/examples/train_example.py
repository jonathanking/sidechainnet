import torch

import sidechainnet
from sidechainnet.utils.data import ProteinDataset
from sidechainnet.examples.models import RGN


def main():
    """ An example of model training. """
    datasets = sidechainnet.load_datasets("../../data/sidechainnet/sidechainnet_casp12_100.pt")
    model = RGN()



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
    main()


