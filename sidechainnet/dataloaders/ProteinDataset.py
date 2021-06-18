"""A class extending torch.utils.data.Dataset for batching protein data."""

import torch.utils

from sidechainnet.utils.sequence import VOCAB, DSSPVocabulary


class ProteinDataset(torch.utils.data.Dataset):
    """This dataset holds lists of sequences, angles, and coordinates for each protein."""

    def __init__(self,
                 scn_data_split,
                 split_name,
                 scn_data_settings,
                 created_on,
                 add_sos_eos=False,
                 sort_by_length=False,
                 reverse_sort=True):

        # Organize data
        self.seqs = [VOCAB.str2ints(s, add_sos_eos) for s in scn_data_split['seq']]
        self.str_seqs = scn_data_split['seq']
        self.angs = scn_data_split['ang']
        self.crds = scn_data_split['crd']
        self.msks = [
            [1 if m == "+" else 0 for m in mask] for mask in scn_data_split['msk']
        ]
        self.evos = scn_data_split['evo']
        self.ids = scn_data_split['ids']
        self.resolutions = scn_data_split['res']
        dssp_vocab = DSSPVocabulary()
        self.secs = [dssp_vocab.str2ints(s, add_sos_eos) for s in scn_data_split['sec']]
        self.mods = scn_data_split['mod']  # Arrays with 1 where non-std residues were standardized

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
                enumerate(self.angs), key=lambda x: x[1].shape[0], reverse=reverse_sort)
        ]
        self.seqs = [self.seqs[i] for i in sorted_len_indices]
        self.str_seqs = [self.str_seqs[i] for i in sorted_len_indices]
        self.angs = [self.angs[i] for i in sorted_len_indices]
        self.crds = [self.crds[i] for i in sorted_len_indices]
        self.msks = [self.msks[i] for i in sorted_len_indices]
        self.evos = [self.evos[i] for i in sorted_len_indices]
        self.ids = [self.ids[i] for i in sorted_len_indices]
        self.resolutions = [self.resolutions[i] for i in sorted_len_indices]
        self.secs = [self.secs[i] for i in sorted_len_indices]
        self.mods = [self.mods[i] for i in sorted_len_indices]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return (self.ids[idx], self.seqs[idx], self.msks[idx], self.evos[idx],
                self.secs[idx], self.angs[idx], self.crds[idx], self.resolutions[idx],
                self.mods[idx], self.str_seqs[idx])

    def __str__(self):
        """Describe this dataset to the user."""
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
