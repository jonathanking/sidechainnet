import numpy as np
import torch.utils

from sidechainnet.utils.download import MAX_SEQ_LEN
from sidechainnet.utils.sequence import VOCAB, ProteinVocabulary


class ProteinDataset(torch.utils.data.Dataset):
    """
    This dataset can hold lists of sequences, angles, and coordinates for
    each protein.
    """

    def __init__(self,
                 scn_data_split,
                 split_name,
                 scn_data_settings,
                 created_on,
                 add_sos_eos=False,
                 sort_by_length=False,
                 reverse_sort=True):

        # Organize data
        self._seqs = [VOCAB.str2ints(s, add_sos_eos) for s in scn_data_split['seq']]
        self._angs = scn_data_split['ang']
        self._crds = scn_data_split['crd']
        self._msks = scn_data_split['msk']
        self._evos = scn_data_split['evo']
        self._ids = scn_data_split['ids']

        # Add metadata
        self.casp_version = scn_data_settings['casp_version']
        self.created_on = created_on
        self.split_name = split_name
        if self.split_name == "train":
            self.thinning = scn_data_settings['thinning']
        else:
            self.thinning = None

        if sort_by_length:
            self._sort_by_length(reverse_sort)

    def _sort_by_length(self, reverse_sort):
        """Sorts all data entries by sequence length."""
        sorted_len_indices = [
            a[0] for a in sorted(
                enumerate(self._angs), key=lambda x: x[1].shape[0], reverse=reverse_sort)
        ]
        self._seqs = [self._angs[i] for i in sorted_len_indices]
        self._angs = [self._angs[i] for i in sorted_len_indices]
        self._crds = [self._crds[i] for i in sorted_len_indices]
        self._msks = [self._msks[i] for i in sorted_len_indices]
        self._evos = [self._evos[i] for i in sorted_len_indices]
        self._ids = [self._ids[i] for i in sorted_len_indices]

    @property
    def n_insts(self):
        """ Property for dataset size """
        return len(self._seqs)

    def __len__(self):
        return self.n_insts

    def __getitem__(self, idx):

        return self._ids[idx], self._seqs[idx], self._angs[idx], self._crds[
            idx], self._evos[idx], self._msks[idx]

    def __str__(self):
        """Describes this dataset to the user."""
        if self.thinning:
            return (f"ProteinDataset(casp_version={self.casp_version}, "
                    f"split='{self.split_name}', "
                    f"thinning={self.thinning}, "
                    f"n_proteins={len(self)}, "
                    f"created='{self.created_on}')")
        else:
            return (f"ProteinDataset(casp_version={self.casp_version}, "
                    f"split='{self.split_name}', "
                    f"n_proteins={len(self)}, "
                    f"created='{self.created_on}')")

    def __repr__(self):
        return self.__str__()
