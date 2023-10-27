"""A class for generating protein sequence/structure batches of similar length."""

import random
import numpy as np
import scipy
import torch
import torch.utils


class SimilarLengthBatchSampler(torch.utils.data.Sampler):
    """Samples a protein dataset by length.

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

    # TODO add support to automatically yield largest batch first for debugging/alloc

    def __init__(self,
                 dataset,
                 batch_size,
                 dynamic_batching,
                 optimize_batch_for_cpus,
                 n_bins=30,
                 downsample=None,
                 use_largest_bin=False,
                 shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.dynamic_batching = dynamic_batching
        self.optimize_batch_for_cpus = optimize_batch_for_cpus
        self.cpu_count = torch.multiprocessing.cpu_count()
        self.downsample = downsample
        self.use_largest_bin = use_largest_bin
        self.shuffle = shuffle
        self._original_shuffle_method = shuffle
        if shuffle:
            self.turn_off_descending()

        equalization_method = 'n_res' if dynamic_batching else 'n_proteins'

        self._init_histogram_bins(n_bins, equalization_method)

    def _init_histogram_bins(self, n_bins, equalize='n_res'):
        assert equalize in [
            'n_res', 'n_proteins'
        ], "Choose to either equalize the number of residues or proteins per bin."
        # Compute length-based histogram bins and probabilities
        self.lens = [len(p) for p in self.dataset]  # Assumes max seq len already applied
        assert all(x <= y for x, y in zip(self.lens, self.lens[1:])), (
            "Data must be in ascending order.")
        total_res_len = sum(self.lens)
        self.bin_map = {}
        self.bin_avg_len = {}
        self.median_len = np.median(self.lens)

        def record_bin(bin_idx, ptn_idx):
            try:
                self.bin_map[bin_idx].append(ptn_idx)
            except KeyError:
                self.bin_map[bin_idx] = [ptn_idx]

        cur_bin = 0
        cur_bin_sum = 0

        if equalize == 'n_res':
            for protein_idx, protein_length in enumerate(self.lens):
                # We're careful not to accidentally make an extra bin for last few items
                if (cur_bin != n_bins - 1 and cur_bin_sum > total_res_len // n_bins):
                    self.bin_avg_len[cur_bin] = cur_bin_sum // len(self.bin_map[cur_bin])
                    cur_bin += 1
                    cur_bin_sum = 0

                record_bin(cur_bin, protein_idx)
                cur_bin_sum += protein_length
            self.bin_avg_len[cur_bin] = cur_bin_sum // len(self.bin_map[cur_bin])

        elif equalize == 'n_proteins':
            ranked = scipy.stats.rankdata(self.lens)
            percentile = ranked / len(self.lens) * 100
            bins_percentile = np.linspace(0, 100, num=n_bins)
            data_binned_indices = np.digitize(percentile, bins_percentile) - 1  # offset
            for protein_idx, protein_bin in enumerate(data_binned_indices):
                record_bin(protein_bin, protein_idx)

    def __len__(self):

        # Since in either case we use simple chunking generators, we can count the number
        # of batches each will yield. Dynamic batching simply changes the size yielded
        # from each generator.

        if self.dynamic_batching:
            batches = 0
            target_num_res = self.median_len * self.batch_size
            for bin_idx, ptn_list in self.bin_map.items():
                effective_batch_size = max(
                    1, int(target_num_res // self.bin_avg_len[bin_idx]))
                batches += np.ceil(len(ptn_list) / effective_batch_size)
            return int(batches)

        else:
            batches = 0
            for bin_idx, ptn_list in self.bin_map.items():
                batches += np.ceil(len(ptn_list) / self.batch_size)
            return int(batches)

    def __iter__(self):
        """Yield indices representing protein batches intelligently by length."""

        def bin_generator(some_list, n):
            """Create a chunking generator to yield fixed-size batches from a bin list."""
            for i in range(0, len(some_list), n):
                yield some_list[i:i + n]

        def make_new_mapping_for_epoch():
            """Create new generators for batching this epoch, shuffling if requested."""
            cur_bin_mapping = {}
            for bin, ptn_list in self.bin_map.items():
                # Set effective batch size
                if self.dynamic_batching:
                    target_num_res = self.median_len * self.batch_size
                    effective_batch_size = max(
                        1, int(target_num_res // self.bin_avg_len[bin]))
                else:
                    effective_batch_size = self.batch_size
                # Shuffle for this epoch if requested
                if self.shuffle:
                    random.shuffle(ptn_list)
                if self.descending:
                    ptn_list = ptn_list[::-1]
                # Initialize generators for each bin
                cur_bin_mapping[bin] = bin_generator(ptn_list, effective_batch_size)
            return cur_bin_mapping

        def batch_generator():
            """Create main batch generator which accesses smaller bin generators."""
            # Re-initialize generators per bin
            bin_generators = make_new_mapping_for_epoch()
            # Select bin, deleting its reference if the generator has run out
            while len(bin_generators) > 0:
                if self.shuffle:
                    selected_bin = np.random.choice(list(bin_generators.keys()))
                else:
                    selected_bin = list(bin_generators.keys())[0]
                try:
                    yield next(bin_generators[selected_bin])
                except StopIteration:
                    del bin_generators[selected_bin]
                    continue

        def desc_batch_generator():
            """Yield largest proteins first."""
            # Re-initialize generators per bin
            bin_generators = make_new_mapping_for_epoch()
            # Select bin, deleting its reference if the generator has run out
            while len(bin_generators) > 0:
                selected_bin = list(bin_generators.keys())[-1]
                try:
                    yield next(bin_generators[selected_bin])
                except StopIteration:
                    del bin_generators[selected_bin]
                    continue

        # TODO add support for batch sizes divisible by cpu count
        # if self.optimize_batch_for_cpus:
        #     largest_possible = int(self.dynamic_batching / self.hist_bins[bin])
        #     this_batch_size = max(1,
        #         largest_possible - (largest_possible % self.cpu_count))
        if self.descending:
            return desc_batch_generator()
        else:
            return batch_generator()

    def make_descending(self):
        """Modify sampling method for future epochs to yield largest proteins first."""
        self.descending = True
        self.shuffle = False

    def turn_off_descending(self):
        """Revert sampling to method before self.make_descending() was called."""
        self.descending = False
        self.shuffle = self._original_shuffle_method


if __name__ == '__main__':
    import sidechainnet as scn
    d = scn.load(12, 30, scn_dataset=True)
    samp = SimilarLengthBatchSampler(d,
                                     24,
                                     dynamic_batching=True,
                                     optimize_batch_for_cpus=False,
                                     shuffle=False)

    samp.make_descending()
    indices = []
    lens = []
    i = 0
    for b in iter(samp):
        print(b)
        indices.extend(b)
        i += 1
        if i == 4:
            break
    print('test')
    samp.turn_off_descending()
    i = 0
    for b in iter(samp):
        print(b)
        indices.extend(b)
        i += 1
        if i == 4:
            break
