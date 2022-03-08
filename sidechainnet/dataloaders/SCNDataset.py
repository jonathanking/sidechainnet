"""Defines high-level objects for interfacing with raw SidechainNet data.

To utilize SCNDataset, pass scn_dataset=True to scn.load().

    >>> d = scn.load("debug", scn_dataset=True)
    >>> d
    SCNDataset(n=461)

SCNDatasets containt SCNProtein objects. See SCNProtein.py for more information.
SCNProteins may be iterated over or selected from the SCNDataset.

    >>> d["1HD1_1_A"]
    SCNProtein(1HD1_1_A, len=75, missing=0, split='train')
"""
import numpy as np
import torch

from sidechainnet.dataloaders.SCNProtein import SCNProtein
from sidechainnet.utils.organize import compute_angle_means


class SCNDataset(torch.utils.data.Dataset):
    """A representation of a SidechainNet dataset."""

    def __init__(self,
                 data,
                 split_name="",
                 trim_edges=False,
                 sort_by_length='ascending') -> None:
        """Initialize a SCNDataset from underlying SidechainNet formatted dictionary."""
        super().__init__()
        # Determine available datasplits
        self.splits = []
        for split_label in ['train', 'valid', 'test']:
            for existing_data_label in data.keys():
                if split_label in existing_data_label:
                    self.splits.append(existing_data_label)

        # If only a single split was provided, prepare the data for protein construction
        if not len(self.splits):
            assert split_name, "Please provide a split name if providing a single split."
            self.splits.append(split_name)
            data = {
                split_name: data,
                "settings": {
                    "angle_means":
                        compute_angle_means(data['ang']) if data['ang'] else None
                }
            }

        # TODO: handle more cleverly the case when no angle/other data is provided

        self.split_to_ids = {}
        self.ids_to_SCNProtein = {}
        self.idx_to_SCNProtein = {}
        _unsorted_proteins = []

        # Create SCNProtein objects and add to data structure
        for split in self.splits:
            d = data[split]
            for c, a, s, u, m, e, n, r, z, i in zip(d['crd'], d['ang'], d['seq'],
                                                    d['ums'], d['msk'], d['evo'],
                                                    d['sec'], d['res'], d['mod'],
                                                    d['ids']):
                try:
                    self.split_to_ids[split].append(i)
                except KeyError:
                    self.split_to_ids[split] = [i]

                p = SCNProtein(coordinates=c,
                               angles=a,
                               sequence=s,
                               unmodified_seq=u,
                               mask=m,
                               evolutionary=e,
                               secondary_structure=n,
                               resolution=r,
                               is_modified=z,
                               id=i,
                               split=split)
                if trim_edges:
                    p.trim_edges()
                self.ids_to_SCNProtein[i] = p
                _unsorted_proteins.append(p)

        if sort_by_length == 'ascending':
            argsorted = np.argsort([len(p) for p in _unsorted_proteins])
        elif sort_by_length == 'descending':
            argsorted = np.argsort([len(p) for p in _unsorted_proteins])[::-1]
        sorted_idx = 0
        for unsorted_idx in argsorted:
            self.idx_to_SCNProtein[sorted_idx] = _unsorted_proteins[unsorted_idx]
            sorted_idx += 1
        del _unsorted_proteins

        # Add metadata
        self.angle_means = compute_angle_means(
            [p.angles for p in self.ids_to_SCNProtein.values()])
        self.lengths = np.array([len(p) for p in self])

    def get_protein_list_by_split_name(self, split_name):
        """Return list of SCNProtein objects belonging to str split_name."""
        return [p for p in self if p.split == split_name]

    def __getitem__(self, id):
        """Retrieve a protein by index or ID (name, e.g. '1A9U_1_A')."""
        if isinstance(id, str):
            return self.ids_to_SCNProtein[id]
        elif isinstance(id, slice):
            step = 1 if not id.step else id.step
            stop = len(self) if not id.stop else id.stop
            start = 0 if not id.start else id.start
            stop = len(self) + stop if stop < 0 else stop
            start = len(self) + start if start < 0 else start
            return [self.idx_to_SCNProtein[i] for i in range(start, stop, step)]
        else:
            return self.idx_to_SCNProtein[id]

    def __len__(self):
        """Return number of proteins in the dataset."""
        return len(self.idx_to_SCNProtein)

    def __iter__(self):
        """Iterate over SCNProtein objects."""
        for i in range(len(self)):
            yield self.idx_to_SCNProtein[i]

    def __repr__(self) -> str:
        """Represent SCNDataset as a string."""
        if len(self.splits) == 1:
            split_str = f", split={self.splits[0]}"
        else:
            split_str = ""
        return f"SCNDataset(n={len(self)}{split_str})"

    def filter_ids(self, to_keep):
        """Remove proteins whose IDs are not included in list to_keep."""
        to_delete = []
        for pnid in self.ids_to_SCNProtein.keys():
            if pnid not in to_keep:
                to_delete.append(pnid)
        for pnid in to_delete:
            p = self.ids_to_SCNProtein[pnid]
            self.split_to_ids[p.split].remove(pnid)
            del self.ids_to_SCNProtein[pnid]
        self.idx_to_SCNProtein = {}
        for i, protein in enumerate(self):
            self.idx_to_SCNProtein[i] = protein

    def _sort_by_length(self, reverse_sort):
        """Sorts all data entries by sequence length."""
        raise NotImplementedError
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
