"""Implements a collating function for use with PyTorch's DataLoaders."""
import collections

import numpy as np
import torch
import torch.utils.data

from sidechainnet.dataloaders.SimilarLengthBatchSampler import SimilarLengthBatchSampler
from sidechainnet.dataloaders.ProteinDataset import ProteinDataset
from sidechainnet.utils.sequence import VOCAB, DSSPVocabulary
from sidechainnet.structure.build_info import NUM_COORDS_PER_RES
from sidechainnet.utils.download import MAX_SEQ_LEN

Batch = collections.namedtuple(
    "Batch", "pids seqs msks evos secs angs "
    "crds int_seqs seq_evo_sec resolutions is_modified "
    "lengths str_seqs")


def get_collate_fn(aggregate_input, seqs_as_onehot=None):
    """Return a collate function for collating ProteinDataset batches.

    Args:
        aggregate_input: Boolean. If true, combine input items (seq, pssms) into a single
            tensor, as opposed to separate tensors.
        seqs_as_onehot: Boolean or None. If None, sequences are represented as one-hot
            vectors during aggregation or represented as integer sequences when not
            aggregated. The user may also specify True if they would like one-hot vectors
            returned iff aggregate_input is False.

    Returns:
        A collate function capable of collating batches from a ProteinDataset.
    """
    if seqs_as_onehot is None:
        if aggregate_input:
            seqs_as_onehot = True
        else:
            seqs_as_onehot = False

    if not seqs_as_onehot and aggregate_input:
        raise ValueError("Sequences must be represented as one-hot vectors if model input"
                         " is to be aggregated.")

    def collate_fn(insts):
        """Collates items extracted from a ProteinDataset, returning all items separately.

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
        pnids, sequences, masks, pssms, secs, angles, coords, resolutions, mods, str_seqs = list(
            zip(*insts))
        lengths = tuple(len(s) for s in sequences)
        max_batch_len = max(lengths)

        int_seqs = pad_for_batch(sequences,
                                 max_batch_len,
                                 'seq',
                                 seqs_as_onehot=False,
                                 vocab=VOCAB)
        padded_seqs = pad_for_batch(sequences,
                                    max_batch_len,
                                    'seq',
                                    seqs_as_onehot=seqs_as_onehot,
                                    vocab=VOCAB)
        padded_secs = pad_for_batch(secs,
                                    max_batch_len,
                                    'seq',
                                    seqs_as_onehot=seqs_as_onehot,
                                    vocab=DSSPVocabulary())
        padded_msks = pad_for_batch(masks, max_batch_len, 'msk')
        padded_pssms = pad_for_batch(pssms, max_batch_len, 'pssm')
        padded_angs = pad_for_batch(angles, max_batch_len, 'ang')
        padded_crds = pad_for_batch(coords, max_batch_len, 'crd')
        padded_mods = pad_for_batch(mods, max_batch_len, 'msk')

        # Non-aggregated model input
        if not aggregate_input:
            return Batch(pids=pnids,
                         seqs=padded_seqs,
                         msks=padded_msks,
                         evos=padded_pssms,
                         secs=padded_secs,
                         angs=padded_angs,
                         crds=padded_crds,
                         int_seqs=int_seqs,
                         seq_evo_sec=None,
                         resolutions=resolutions,
                         is_modified=padded_mods,
                         lengths=lengths,
                         str_seqs=str_seqs)

        # Aggregated model input
        elif aggregate_input:
            seq_evo_sec = torch.cat(
                [padded_seqs.float(), padded_pssms,
                 padded_secs.float()], dim=-1)

            return Batch(pids=pnids,
                         seqs=padded_seqs,
                         msks=padded_msks,
                         evos=padded_pssms,
                         secs=padded_secs,
                         angs=padded_angs,
                         crds=padded_crds,
                         int_seqs=int_seqs,
                         seq_evo_sec=seq_evo_sec,
                         resolutions=resolutions,
                         is_modified=padded_mods,
                         lengths=lengths,
                         str_seqs=str_seqs)

    return collate_fn


def pad_for_batch(items, batch_length, dtype="", seqs_as_onehot=False, vocab=None):
    """Pad a list of items to batch_length using values dependent on the item type.

    Args:
        items: List of items to pad (i.e. sequences or masks represented as arrays of
            numbers, angles, coordinates, pssms).
        batch_length: The integer maximum length of any of the items in the input. All
            items are padded so that their length matches this number.
        dtype: A string ('seq', 'msk', 'pssm', 'ang', 'crd') reperesenting the type of
            data included in items.
        seqs_as_onehot: Boolean. If True, sequence-type data will be returned in 1-hot
            vector form.
        vocab: DSSPVocabulary or ProteinVocabulary. Vocabulary object for translating
            and handling sequence-type data.

    Returns:
         A padded list of the input items, all independently converted to Torch tensors.
    """
    batch = []
    if dtype == "seq":
        # Sequences are padded with a specific VOCAB pad character
        for seq in items:
            z = np.ones((batch_length - len(seq))) * vocab.pad_id
            c = np.concatenate((seq, z), axis=0)
            batch.append(c)
        batch = np.array(batch)
        batch = batch[:, :MAX_SEQ_LEN]
        batch = torch.LongTensor(batch)
        if seqs_as_onehot:
            batch = torch.nn.functional.one_hot(batch, len(vocab))
            if vocab.include_pad_char:
                # Delete the column for the pad character since it is implied (0-vector)
                if len(batch.shape) == 3:
                    batch = batch[:, :, :-1]
                elif len(batch.shape) == 2:
                    batch = batch[:, :-1]
                else:
                    raise ValueError(f"Unexpected batch dimension {str(batch.shape)}.")
    elif dtype == "msk":
        # Mask sequences (1 if present, 0 if absent) are padded with 0s
        for msk in items:
            z = np.zeros((batch_length - len(msk)))
            c = np.concatenate((msk, z), axis=0)
            batch.append(c)
        batch = np.array(batch)
        batch = batch[:, :MAX_SEQ_LEN]
        batch = torch.LongTensor(batch)
    elif dtype in ["pssm", "ang"]:
        # Mask other features with 0-vectors of a matching shape
        for item in items:
            z = np.zeros((batch_length - len(item), item.shape[-1]))
            c = np.concatenate((item, z), axis=0)
            batch.append(c)
        batch = np.array(batch)
        batch = batch[:, :MAX_SEQ_LEN]
        batch = torch.FloatTensor(batch)
    elif dtype == "crd":
        for item in items:
            z = np.zeros((batch_length * NUM_COORDS_PER_RES - len(item), item.shape[-1]))
            c = np.concatenate((item, z), axis=0)
            batch.append(c)
        batch = np.array(batch)
        # There are multiple rows per res, so we allow the coord matrix to be larger
        batch = batch[:, :MAX_SEQ_LEN * NUM_COORDS_PER_RES]
        batch = torch.FloatTensor(batch)

    return batch


def prepare_dataloaders(data,
                        aggregate_model_input,
                        collate_fn=None,
                        batch_size=32,
                        num_workers=1,
                        seq_as_onehot=None,
                        dynamic_batching=True,
                        optimize_for_cpu_parallelism=False,
                        train_eval_downsample=0.1):
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
        collate_fn = get_collate_fn(aggregate_model_input, seqs_as_onehot=seq_as_onehot)

    train_dataset = ProteinDataset(data['train'], 'train', data['settings'], data['date'])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=num_workers,
        collate_fn=collate_fn,
        batch_sampler=SimilarLengthBatchSampler(
            train_dataset,
            batch_size,
            dynamic_batch=batch_size *
            data['settings']['lengths'].mean() if dynamic_batching else None,
            optimize_batch_for_cpus=optimize_for_cpu_parallelism,
        ))

    train_eval_loader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=num_workers,
        collate_fn=collate_fn,
        batch_sampler=SimilarLengthBatchSampler(
            train_dataset,
            batch_size,
            dynamic_batch=None,
            optimize_batch_for_cpus=optimize_for_cpu_parallelism,
            downsample=train_eval_downsample))

    valid_loaders = {}
    valid_splits = [splitname for splitname in data.keys() if "valid" in splitname]
    for vsplit in valid_splits:
        try:
            valid_loader = torch.utils.data.DataLoader(ProteinDataset(
                data[vsplit],
                vsplit,
                data['settings'],
                data['date']),
                                                       num_workers=1,
                                                       batch_size=batch_size,
                                                       collate_fn=collate_fn)
            valid_loaders[vsplit] = valid_loader
        except KeyError:
            pass

    test_loader = torch.utils.data.DataLoader(ProteinDataset(data['test'], 'test',
                                                             data['settings'],
                                                             data['date']),
                                              num_workers=num_workers,
                                              batch_size=batch_size,
                                              collate_fn=collate_fn)

    dataloaders = {
        'train': train_loader,
        'train-eval': train_eval_loader,
        'test': test_loader
    }
    dataloaders.update(valid_loaders)

    return dataloaders


def get_dataloader_from_dataset_dict(dictionary):
    collate_fn = get_collate_fn(False, seqs_as_onehot=True)

    train_dataset = ProteinDataset(data['train'], 'train', data['settings'], data['date'])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=num_workers,
        collate_fn=collate_fn,
        batch_sampler=SimilarLengthBatchSampler(
            train_dataset,
            batch_size,
            dynamic_batch=batch_size *
            data['settings']['lengths'].mean() if dynamic_batching else None,
            optimize_batch_for_cpus=optimize_for_cpu_parallelism,
        ))
