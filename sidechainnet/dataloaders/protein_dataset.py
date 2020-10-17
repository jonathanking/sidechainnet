import numpy as np
import torch.utils

from sidechainnet.dataloaders.collate import MAX_SEQ_LEN
from sidechainnet.utils.sequence import VOCAB, ProteinVocabulary


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
        assert (angs is None) or (len(seqs) == len(angs) and len(angs) == len(crds))
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

    def __init__(self, seqs=None, angs=None, crds=None, add_sos_eos=False, bins="auto"):

        assert seqs is not None
        assert (angs is None) or (len(seqs) == len(angs) and len(angs) == len(crds))
        self.vocab = ProteinVocabulary()
        self._seqs, self._angs, self._crds = [], [], []
        for i in range(len(seqs)):
            self._seqs.append(VOCAB.str2ints(seqs[i], add_sos_eos))
            self._angs.append(angs[i])
            self._crds.append(crds[i])

        # Compute length-based histogram bins and probabilities
        self.lens = list(
            map(lambda x: len(x) if len(x) <= MAX_SEQ_LEN else MAX_SEQ_LEN, self._seqs))
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
