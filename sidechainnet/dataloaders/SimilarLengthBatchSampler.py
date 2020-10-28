"""A class for generating protein sequence/structure batches of similar length."""

import numpy as np
import torch
import torch.utils
from sidechainnet.utils.download import MAX_SEQ_LEN


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
                 bins='auto',
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

        self._init_histogram_bins(bins)

    def _init_histogram_bins(self, bins):
        # Compute length-based histogram bins and probabilities
        self.lens = []
        for s in self.data_source.seqs:
            if len(s) <= MAX_SEQ_LEN:
                self.lens.append(len(s))
            else:
                self.lens.append(MAX_SEQ_LEN)

        self.hist_counts, self.hist_bins = np.histogram(self.lens, bins=bins)
        # Make each bin define the rightmost value in each bin, ie '( , ]'.
        self.hist_bins = self.hist_bins[1:]
        self.bin_probs = self.hist_counts / self.hist_counts.sum()
        self.bin_map = {}

        # Compute a mapping from bin number to a list of dataset indices, i.e.:
        #                        { 0: [0, 1, ... 67],
        #                          1: [68, 69, ... 98],
        #                          ...
        #                          }
        # In the batch generation procedure, a sequence length bin is drawn first,
        # then proteins are selected at random from the list of indices. Each bin has
        # proteins of similar length.
        seq_i = 0
        bin_j = 0
        while seq_i < len(self.data_source.seqs):
            if self.lens[seq_i] <= self.hist_bins[bin_j]:
                try:
                    self.bin_map[bin_j].append(seq_i)
                except KeyError:
                    self.bin_map[bin_j] = [seq_i]
                seq_i += 1
            else:
                bin_j += 1

    def __len__(self):
        # If batches are dynamically sized to contain the same number of
        # residues then the approximate number of batches is the total number
        # of residues in the dataset divided by the size of the dynamic batch.

        if self.dynamic_batch:
            numerator = sum(self.lens) * self.repeat_train
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
                    bin = len(self.hist_bins) - 1
                else:
                    bin = np.random.choice(range(len(self.hist_bins)), p=self.bin_probs)
                if self.dynamic_batch:
                    # Make the batch size a multiple of the number of available
                    # CPUs for fast DRMSD loss computation
                    if self.optimize_batch_for_cpus:
                        largest_possible = int(self.dynamic_batch / self.hist_bins[bin])
                        this_batch_size = max(
                            1, largest_possible - (largest_possible % self.cpu_count))
                    else:
                        this_batch_size = max(
                            1, int(self.dynamic_batch / self.hist_bins[bin]))
                else:
                    this_batch_size = self.batch_size
                yield np.random.choice(self.bin_map[bin], size=this_batch_size)
                i += 1

        return batch_generator()
