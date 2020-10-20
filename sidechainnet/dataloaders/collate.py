import numpy as np
import torch
import torch.utils.data

from sidechainnet.dataloaders.SimilarLengthBatchSampler import SimilarLengthBatchSampler
from sidechainnet.dataloaders.ProteinDataset import ProteinDataset
from sidechainnet.utils.sequence import VOCAB
from sidechainnet.structure.build_info import NUM_COORDS_PER_RES
from sidechainnet.utils.download import VALID_SPLITS, MAX_SEQ_LEN


def unaggregated_collate_fn(insts):
    """Collates items extracted from a ProteinDataset, returning all items separately.
    
    Args:
        insts: A list of tuples, each containing one pnid, sequence, mask, pssm, angle, 
            and coordinate extracted from a corresponding ProteinDataset.
    
    Returns:
        A tuple of the same information provided in the input, where each data type has
        been extracted in a list of its own. In other words, the returned tuple has one
        list for each of (pnids, seqs, msks, pssms, angs, crds). Each item in each list
        is padded to the maximum length of sequences in the batch.
             
    """
    # Instead of working with a list of tuples, we extract out each category of info
    # so it can be padded and re-provided to the user.
    pnids, sequences, masks, pssms, angles, coords, = list(zip(*insts))
    max_batch_len = max(len(s) for s in sequences)

    padded_seqs = pad_for_batch(sequences, max_batch_len, 'seq')
    padded_msks = pad_for_batch(masks, max_batch_len, 'msk')
    padded_pssms = pad_for_batch(pssms, max_batch_len, 'pssm')
    padded_angs = pad_for_batch(angles, max_batch_len, 'ang')
    padded_crds = pad_for_batch(coords, max_batch_len, 'crd')

    return pnids, padded_seqs, padded_msks, padded_pssms, padded_angs, padded_crds


def pad_for_batch(items, batch_length, type=""):
    """Pads a list of items to batch_length using values dependent on the item type.
    
    Args:
        items: List of items to pad (i.e. sequences or masks represented as arrays of 
            numbers, angles, coordinates, pssms).
        batch_length: The integer maximum length of any of the items in the input. All 
            items are padded so that their length matches this number.
        type: A string ('seq', 'msk', 'pssm', 'ang', 'crd') reperesenting the type of
            data included in items.
    
    Returns:
         A padded list of the input items, all independently converted to Torch tensors.
    """
    batch = []
    if type == "seq":
        # Sequences are padded with a specific VOCAB pad character
        for seq in items:
            z = np.ones((batch_length - len(seq))) * VOCAB.pad_id
            c = np.concatenate((seq, z), axis=0)
            batch.append(c)
        batch = np.array(batch)
        batch = batch[:, :MAX_SEQ_LEN]
        batch = torch.LongTensor(batch)
    elif type == "msk":
        # Mask sequences (1 if present, 0 if absent) are padded with 0s
        for msk in items:
            z = np.zeros((batch_length - len(msk)))
            c = np.concatenate((msk, z), axis=0)
            batch.append(c)
        batch = np.array(batch)
        batch = batch[:, :MAX_SEQ_LEN]
        batch = torch.LongTensor(batch)
    elif type in ["pssm", "ang"]:
        # Mask other features with 0-vectors of a matching shape
        for item in items:
            z = np.zeros((batch_length - len(item), item.shape[-1]))
            c = np.concatenate((item, z), axis=0)
            batch.append(c)
        batch = np.array(batch)
        batch = batch[:, :MAX_SEQ_LEN]
        batch = torch.FloatTensor(batch)
    elif type == "crd":
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
                        batch_size=32,
                        num_workers=1,
                        optimize_for_cpu_parallelism=False,
                        train_eval_downsample=0.1):
    """
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
    if aggregate_model_input:
        raise NotImplementedError
    else:
        collate_fn = unaggregated_collate_fn

    train_dataset = ProteinDataset(data['train'], 'train', data['settings'], data['date'])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=num_workers,
        collate_fn=collate_fn,
        batch_sampler=SimilarLengthBatchSampler(
            train_dataset,
            batch_size,
            dynamic_batch=batch_size * MAX_SEQ_LEN,
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
    for split in VALID_SPLITS:
        valid_loader = torch.utils.data.DataLoader(ProteinDataset(
            data[f'valid-{split}'], f'valid-{split}', data['settings'], data['date']),
                                                   num_workers=num_workers,
                                                   batch_size=batch_size,
                                                   collate_fn=collate_fn)
        valid_loaders[split] = valid_loader

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
    dataloaders.update({f'valid-{split}': valid_loaders[split] for split in VALID_SPLITS})

    return dataloaders


if __name__ == "__main__":
    import sidechainnet as scn
    from sidechainnet.dataloaders.ProteinDataset import ProteinDataset
    d = scn.load(12,
                 30,
                 scn_dir="/home/jok120/dev_sidechainnet/data/sidechainnet",
                 with_pytorch="dataloaders",
                 aggregate_model_input=False,
                 batch_size=4,
                 num_workers=1)
    for batch in d['train']:
        print(batch)
        break
