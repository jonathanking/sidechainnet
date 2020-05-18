import numpy as np
import torch
import torch.utils.data

from sidechainnet.utils.sequence import ProteinVocabulary, VOCAB
from sidechainnet.utils.build_info import NUM_PREDICTED_COORDS

VALID_SPLITS = [10, 20, 30, 40, 50, 70, 90]
MAX_SEQ_LEN = 500


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
        batch = batch[:, :max_seq_len * NUM_PREDICTED_COORDS]
    else:
        batch = batch[:, :max_seq_len]

    if sequences:
        batch = torch.LongTensor(batch)
    else:
        batch = torch.FloatTensor(batch)

    return batch


class ProteinDataset(torch.utils.data.Dataset):
    """
    This dataset can hold lists of sequences, angles, and coordinates for
    each protein.
    """

    def __init__(self,
                 seqs=None,
                 angs=None,
                 crds=None,
                 add_sos_eos=True,
                 sort_by_length=True,
                 reverse_sort=True):

        assert seqs is not None
        assert (angs is None) or (len(seqs) == len(angs) and
                                  len(angs) == len(crds))
        self._seqs, self._angs, self._crds = [], [], []
        for i in range(len(seqs)):
            self._seqs.append(VOCAB.str2ints(seqs[i], add_sos_eos))
            self._angs.append(angs[i])
            self._crds.append(crds[i])

        if sort_by_length:
            sorted_len_indices = [
                a[0] for a in sorted(enumerate(self._angs),
                                     key=lambda x: x[1].shape[0],
                                     reverse=reverse_sort)
            ]
            new_seqs = [self._seqs[i] for i in sorted_len_indices]
            self._seqs = new_seqs
            new_angs = [self._angs[i] for i in sorted_len_indices]
            self._angs = new_angs
            new_crds = [self._crds[i] for i in sorted_len_indices]
            self._crds = new_crds

    @property
    def n_insts(self):
        """ Property for dataset size """
        return len(self._seqs)

    def __len__(self):
        return self.n_insts

    def __getitem__(self, idx):
        if self._angs is not None:
            return self._seqs[idx], self._angs[idx], self._crds[idx]
        return self._seqs[idx]


class BinnedProteinDataset(torch.utils.data.Dataset):
    """
    This dataset can hold lists of sequences, angles, and coordinates for
    each protein.

    Assumes protein data is sorted from shortest to longest (ascending).
    """

    def __init__(self,
                 seqs=None,
                 angs=None,
                 crds=None,
                 add_sos_eos=True,
                 bins="auto"):

        assert seqs is not None
        assert (angs is None) or (len(seqs) == len(angs) and
                                  len(angs) == len(crds))
        self.vocab = ProteinVocabulary()
        self._seqs, self._angs, self._crds = [], [], []
        for i in range(len(seqs)):
            self._seqs.append(VOCAB.str2ints(seqs[i], add_sos_eos))
            self._angs.append(angs[i])
            self._crds.append(crds[i])

        # Compute length-based histogram bins and probabilities
        self.lens = list(
            map(lambda x: len(x)
                if len(x) <= MAX_SEQ_LEN else MAX_SEQ_LEN, self._seqs))
        self.hist_counts, self.hist_bins = np.histogram(self.lens, bins=bins)
        self.hist_bins = self.hist_bins[
            1:]  # make each bin define the rightmost value in each bin, ie '( , ]'.
        self.bin_probs = self.hist_counts / self.hist_counts.sum()
        self.bin_map = {}

        # Compute a mapping from bin number to index in dataset
        seq_i = 0
        bin_j = 0
        while seq_i < len(self._seqs):
            if self.lens[seq_i] <= self.hist_bins[bin_j]:
                try:
                    self.bin_map[bin_j].append(seq_i)
                except KeyError:
                    self.bin_map[bin_j] = [seq_i]
                seq_i += 1
            else:
                bin_j += 1

    @property
    def n_insts(self):
        """ Property for dataset size """
        return len(self._seqs)

    def __len__(self):
        return self.n_insts

    def __getitem__(self, idx):
        if self._angs is not None:
            return self._seqs[idx], self._angs[idx], self._crds[idx]
        return self._seqs[idx]


class SimilarLengthBatchSampler(torch.utils.data.Sampler):
    """
    When turned into an iterator, this Sampler is designed to yield batches
    of indices at a time. The indices correspond to sequences in the dataset,
    grouped by sequence length. This has the effect of yielding batches where
    all items in a batch have similar length, but the average length of any
    particular batch is completely random.

    When optimize_batch_for_cpus is True, the sampler will always yield batches
    that are a multiple of the number of available CPUs.

    When downsample is a float, the dataset is effectively downsampled by that fraction.
    i.e. if downsample = 0.3, then about 30% of the dataset is used.
    """

    def __init__(self,
                 data_source,
                 batch_size,
                 dynamic_batch,
                 optimize_batch_for_cpus,
                 downsample=None,
                 use_largest_bin=False,
                 repeat_train=None):
        self.data_source = data_source
        self.batch_size = batch_size
        self.dynamic_batch = dynamic_batch
        self.optimize_batch_for_cpus = optimize_batch_for_cpus
        self.cpu_count = torch.multiprocessing.cpu_count()
        self.downsample = downsample
        self.use_largest_bin = use_largest_bin
        self.repeat_train = repeat_train if repeat_train else 1

    def __len__(self):
        # If batches are dynamically sized to contain the same number of
        # residues then the approximate number of batches is the total number
        # of residues in the dataset divided by the size of the dynamic batch.

        if self.dynamic_batch:
            numerator = sum(self.data_source.lens) * self.repeat_train
            divisor = self.dynamic_batch
        else:
            numerator = len(self.data_source) * self.repeat_train
            divisor = self.batch_size

        if self.downsample:
            numerator *= self.downsample

        return int(np.ceil(numerator / divisor))

    def __iter__(self):

        def batch_generator():
            i = 0
            while i < len(self):
                if self.use_largest_bin:
                    bin = len(self.data_source.hist_bins) - 1
                else:
                    bin = np.random.choice(range(len(
                        self.data_source.hist_bins)),
                                           p=self.data_source.bin_probs)
                if self.dynamic_batch:
                    # Make the batch size a multiple of the number of available
                    # CPUs for fast DRMSD loss computation
                    if self.optimize_batch_for_cpus:
                        largest_possible = int(self.dynamic_batch /
                                               self.data_source.hist_bins[bin])
                        this_batch_size = max(
                            1, largest_possible -
                            (largest_possible % self.cpu_count))
                    else:
                        this_batch_size = max(
                            1,
                            int(self.dynamic_batch /
                                self.data_source.hist_bins[bin]))
                else:
                    this_batch_size = self.batch_size
                yield np.random.choice(self.data_source.bin_map[bin],
                                       size=this_batch_size)
                i += 1

        return batch_generator()


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
    n_bins = "auto"
    train_dataset = BinnedProteinDataset(seqs=data['train']['seq'],
                                         crds=data['train']['crd'],
                                         angs=data['train']['ang'],
                                         add_sos_eos=False,
                                         bins=n_bins)

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
            seqs=data[f'valid-{split}']['seq'],
            crds=data[f'valid-{split}']['crd'],
            angs=data[f'valid-{split}']['ang'],
            add_sos_eos=False),
                                                   num_workers=num_workers,
                                                   batch_size=batch_size,
                                                   collate_fn=paired_collate_fn)
        valid_loaders[split] = valid_loader

    test_loader = torch.utils.data.DataLoader(ProteinDataset(
        seqs=data['test']['seq'],
        crds=data['test']['crd'],
        angs=data['test']['ang'],
        add_sos_eos=False),
                                              num_workers=num_workers,
                                              batch_size=batch_size,
                                              collate_fn=paired_collate_fn)

    return train_loader, train_eval_loader, valid_loaders, test_loader
