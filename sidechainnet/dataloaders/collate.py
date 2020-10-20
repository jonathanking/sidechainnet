import numpy as np
import torch
import torch.utils.data

from sidechainnet.dataloaders.SimilarLengthBatchSampler import SimilarLengthBatchSampler
from sidechainnet.dataloaders.ProteinDataset import ProteinDataset, BinnedProteinDataset
from sidechainnet.utils.sequence import VOCAB
from sidechainnet.structure.build_info import NUM_COORDS_PER_RES
from sidechainnet.utils.download import VALID_SPLITS, MAX_SEQ_LEN


def paired_collate_fn(insts):
    """
    This function creates mini-batches (3-tuples) of sequence, angle and
    coordinate Tensors. insts is a list of tuples, each containing one src
    seq, and target angles and coordindates.
    """
    sequences, angles, coords = list(zip(*insts))
    sequences = collate_fn(sequences, sequences=True, max_seq_len=MAX_SEQ_LEN)
    angles = collate_fn(angles, max_seq_len=MAX_SEQ_LEN)
    coords = collate_fn(coords, coords=True, max_seq_len=MAX_SEQ_LEN)
    return sequences, angles, coords


def collate_fn(insts, coords=False, sequences=False, max_seq_len=None):
    """
    Given a list of tuples to be stitched together into a batch, this function
    pads each instance to the max seq length in batch and returns a batch
    Tensor.
    """
    max_batch_len = max(len(inst) for inst in insts)
    batch = []
    for inst in insts:
        if sequences:
            z = np.ones((max_batch_len - len(inst))) * VOCAB.pad_id
        else:
            z = np.zeros((max_batch_len - len(inst), inst.shape[-1]))
        c = np.concatenate((inst, z), axis=0)
        batch.append(c)
    batch = np.array(batch)

    # Trim batch to be less than maximum length
    if coords:
        batch = batch[:, :max_seq_len * NUM_COORDS_PER_RES]
    else:
        batch = batch[:, :max_seq_len]

    if sequences:
        batch = torch.LongTensor(batch)
    else:
        batch = torch.FloatTensor(batch)

    return batch


def prepare_dataloaders(data,
                        batch_size,
                        num_workers=1,
                        optimize_for_cpu_parallelism=False,
                        train_eval_downsample=0.1):
    """
    Using the pre-processed data, stored in a nested Python dictionary, this
    function returns train, validation, and test set dataloaders with 2 workers
    each. Note that there are multiple validation sets in ProteinNet.
    """
    train_dataset = ProteinDataset(data['train'], 'train', data['settings'], data['date'])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=num_workers,
        collate_fn=paired_collate_fn,
        batch_sampler=SimilarLengthBatchSampler(
            train_dataset,
            batch_size,
            dynamic_batch=batch_size * MAX_SEQ_LEN,
            optimize_batch_for_cpus=optimize_for_cpu_parallelism,
        ))

    train_eval_loader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=num_workers,
        collate_fn=paired_collate_fn,
        batch_sampler=SimilarLengthBatchSampler(
            train_dataset,
            batch_size,
            dynamic_batch=None,
            optimize_batch_for_cpus=optimize_for_cpu_parallelism,
            downsample=train_eval_downsample))

    valid_loaders = {}
    for split in VALID_SPLITS:
        valid_loader = torch.utils.data.DataLoader(ProteinDataset(
            data[f'valid-{split}'], f'valid-{split}', data['settings'], data['date']),
                                                   num_workers=num_workers,
                                                   batch_size=batch_size,
                                                   collate_fn=paired_collate_fn)
        valid_loaders[split] = valid_loader

    test_loader = torch.utils.data.DataLoader(ProteinDataset(data['test'], 'test',
                                                             data['settings'],
                                                             data['date']),
                                              num_workers=num_workers,
                                              batch_size=batch_size,
                                              collate_fn=paired_collate_fn)

    return train_loader, train_eval_loader, valid_loaders, test_loader
