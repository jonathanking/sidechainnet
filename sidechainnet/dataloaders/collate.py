"""Implements a collating function for use with PyTorch's DataLoaders."""

import numpy as np
import torch
import torch.utils.data
from sidechainnet.dataloaders.ProteinBatch import ProteinBatch
from sidechainnet.dataloaders.SCNDataset import SCNDataset

from sidechainnet.dataloaders.SimilarLengthBatchSampler import SimilarLengthBatchSampler


def protein_batch_collate_fn(protein_objects):
    """Collates items extracted from a SCNDataset, returning a ProteinBatch.

    Args:
        insts: A list of tuples, each containing one pnid, sequence, mask, pssm,
            angle, and coordinate extracted from a corresponding ProteinDataset.
        aggregate_input: A boolean that, if True, aggregates the 'model input'
            components of the data, or the data items (seqs, pssms) from which
            coordinates and angles are predicted.

    Returns:
        A tuple of the same information provided in the input, where each data type
        has been extracted in a list of its own. In other words, the returned tuple
        has one list for each of (pnids, seqs, msks, pssms, angs, crds). Each item in
        each list is padded to the maximum length of sequences in the batch.
    """
    # Instead of working with a list of tuples, we extract out each category of info
    # so it can be padded and re-provided to the user.
    pb = ProteinBatch(protein_objects)
    return pb


def prepare_dataloaders(data,
                        aggregate_model_input,
                        collate_fn=None,
                        batch_size=32,
                        num_workers=None,
                        seq_as_onehot=None,
                        dynamic_batching=True,
                        optimize_for_cpu_parallelism=False,
                        train_eval_downsample=0.1,
                        shuffle=True,
                        overfit_batches=0,
                        overfit_batches_small=True,
                        complete_structures_only=False):
    """Return dataloaders for model training according to user specifications.

    Using the pre-processed data, stored in a nested Python dictionary, this
    function returns train, validation, and test set dataloaders with 2 workers
    each. Note that there are multiple validation sets in ProteinNet.

    Args:
        data: A dictionary storing the entirety of a SidechainNet version (i.e. CASP 12).
            It must be organized in the manner described in this code's README, or in the
            paper.
        aggregate_model_input: A boolean that, if True, yields batches of (protein_id,
            model_input, true_angles, true_coordinates) when iterating over the returned
            PyTorch DataLoader. If False, this expands the model_input variable into
            its components (sequence, mask pssm).
        batch_size: Batch size to use when yielding batches from a DataLoader.
    """
    from sidechainnet.utils.download import VALID_SPLITS
    if collate_fn is None:
        collate_fn = protein_batch_collate_fn

    train_dataset = SCNDataset(data['train'],
                               split_name='train',
                               trim_edges=True,
                               sort_by_length='ascending',
                               overfit_batches=overfit_batches,
                               overfit_batches_small=overfit_batches_small,
                               complete_structures_only=complete_structures_only)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=num_workers,
        collate_fn=collate_fn,
        batch_sampler=SimilarLengthBatchSampler(
            train_dataset,
            batch_size,
            dynamic_batching=dynamic_batching,
            optimize_batch_for_cpus=optimize_for_cpu_parallelism,
            shuffle=shuffle))
    train_eval_loader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=num_workers,
        collate_fn=collate_fn,
        batch_sampler=SimilarLengthBatchSampler(
            train_dataset,
            batch_size,
            dynamic_batching=dynamic_batching,
            optimize_batch_for_cpus=optimize_for_cpu_parallelism,
            downsample=train_eval_downsample,
            shuffle=shuffle))

    if dynamic_batching:
        print(f"Approximating {batch_size * np.median(train_dataset.lengths):.0f}"
              " residues/batch.")

    vsplits = filter(lambda split: "valid" in split, data.keys())

    valid_loaders = {}
    for vsplit in vsplits:
        valid_loader = torch.utils.data.DataLoader(SCNDataset(data[vsplit],
                                                              split_name=vsplit,
                                                              trim_edges=True,
                                                              sort_by_length='ascending'),
                                                   num_workers=num_workers,
                                                   batch_size=batch_size,
                                                   collate_fn=collate_fn)
        valid_loaders[vsplit] = valid_loader

    test_loader = torch.utils.data.DataLoader(SCNDataset(data['test'],
                                                         split_name='test',
                                                         trim_edges=True,
                                                         sort_by_length='ascending'),
                                              num_workers=num_workers,
                                              batch_size=batch_size,
                                              collate_fn=collate_fn)

    dataloaders = {
        'train': train_loader,
        'train-eval': train_eval_loader,
        'test': test_loader
    }
    dataloaders.update({vsplit: valid_loaders[vsplit] for vsplit in vsplits})

    return dataloaders
