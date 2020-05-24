import numpy as np
import torch
import torch.utils


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