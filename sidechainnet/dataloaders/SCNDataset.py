"""Defines high-level objects for interfacing with raw SidechainNet data.

To utilize SCNDataset, pass scn_dataset=True to scn.load().

    >>> d = scn.load("debug", scn_dataset=True)
    >>> d
    SCNDataset(n=461)

SCNDatasets contain SCNProtein objects. See SCNProtein.py for more information.
SCNProteins may be iterated over or selected from the SCNDataset.

    >>> d["1HD1_1_A"]
    SCNProtein(1HD1_1_A, len=75, missing=0, split='train')
"""
import copy
import datetime
import os
import pickle
import numpy as np
import torch
import tqdm

from sidechainnet.dataloaders.SCNProtein import SCNProtein
from sidechainnet.utils.organize import compute_angle_means, EMPTY_SPLIT_DICT


class SCNDataset(torch.utils.data.Dataset):
    """A representation of a SidechainNet dataset."""

    def __init__(self,
                 data,
                 split_name="",
                 trim_edges=False,
                 sort_by_length='ascending',
                 overfit_batches=0,
                 overfit_batches_small=True,
                 complete_structures_only=False) -> None:
        """Initialize a SCNDataset from underlying SidechainNet formatted dictionary."""
        super().__init__()
        # Determine available datasplits
        self.splits = []
        for split_label in ['train', 'valid', 'test']:
            for existing_data_label in data.keys():
                if existing_data_label is not None and split_label in existing_data_label:
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

        starting_length = sum([len(data[split]['seq']) for split in self.splits])

        # TODO: handle more cleverly the case when no angle/other data is provided

        self.split_to_ids = {}
        self.ids_to_SCNProtein = {}
        self.idx_to_SCNProtein = {}
        _unsorted_proteins = []

        # Create SCNProtein objects and add to data structure
        for split in self.splits:
            d = data[split]
            count = 0
            for c, a, s, u, m, e, n, r, z, i in zip(d['crd'], d['ang'], d['seq'],
                                                    d['ums'], d['msk'], d['evo'],
                                                    d['sec'], d['res'], d['mod'],
                                                    d['ids']):
                # This portion is simply to skip over very small proteins when ovrftng
                if (not overfit_batches_small and overfit_batches and split == 'train' and
                        count < len(d['seq']) // 2):
                    count += 1
                    continue

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
                if complete_structures_only and "-" in p.mask:
                    continue

                p.trim_to_max_seq_len()  # TODO add an option to not do this by default

                self.ids_to_SCNProtein[i] = p
                _unsorted_proteins.append(p)
                try:
                    self.split_to_ids[split].append(i)
                except KeyError:
                    self.split_to_ids[split] = [i]

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

        # Report excluded entries
        if complete_structures_only:
            n_filtered_entries = starting_length - len(self.lengths)
            print(f"{n_filtered_entries} ({n_filtered_entries/starting_length:.1%})"
                  " data set entries were excluded due to missing residues.")

    @classmethod
    def from_scnproteins(cls, proteins):
        """Create a SCNDataset from a list of SCNProtein objects."""
        data = {}
        for p in proteins:
            if p.split not in data:
                data[p.split] = {
                    'crd': [],
                    'ang': [],
                    'seq': [],
                    'ums': [],
                    'msk': [],
                    'evo': [],
                    'sec': [],
                    'res': [],
                    'mod': [],
                    'ids': []
                }
            data[p.split]['crd'].append(p.coords)
            data[p.split]['ang'].append(p.angles)
            data[p.split]['seq'].append(p.seq)
            data[p.split]['ums'].append(p.unmodified_seq)
            data[p.split]['msk'].append(p.mask)
            data[p.split]['evo'].append(p.evolutionary)
            data[p.split]['sec'].append(p.secondary_structure)
            data[p.split]['res'].append(p.resolution)
            data[p.split]['mod'].append(p.is_modified)
            data[p.split]['ids'].append(p.id)
        return cls(data)

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
            if id < 0:
                id = len(self) + id
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

    def __contains__(self, pnid):
        """Return True if SidechainNet/ProteinNet pnid is in the dataset."""
        return pnid in self.ids_to_SCNProtein

    def delete_ids(self, to_delete):
        """Remove proteins whose IDs are in list to_delete."""
        for pnid in to_delete:
            try:
                p = self.ids_to_SCNProtein[pnid]
                self.split_to_ids[p.split].remove(pnid)
                del self.ids_to_SCNProtein[pnid]
            except KeyError:
                continue  # If the protein is not in the dataset, do nothing
        self.idx_to_SCNProtein = {}
        for i, protein in enumerate(self.ids_to_SCNProtein.values()):
            self.idx_to_SCNProtein[i] = protein

    def filter(self, func, verbose=True):
        """Filter the SCNDataset, keeping all entries where func(protein) is True.

        Args:
            func (function): A function that takes as input a single SCNProtein and
            returns True or False.
        """
        starting_size = len(self)
        to_keep = [p.id for p in filter(func, self)]
        self.filter_ids(to_keep)
        if verbose:
            n_filtered_entries = starting_size - len(self)
            print(f"{n_filtered_entries} ({n_filtered_entries/starting_size:.1%})"
                  " data set entries were excluded by user-defined function.")

    def _sort_by_length(self, reverse_sort):
        """Sorts all data entries by sequence length."""
        raise NotImplementedError

    def pickle(self, path, description=None):
        """Create and save a pickled Python dictionary representing the dataset.

        Args:
            path (str): Path to new file.
        """
        complete_dict = {
            "date": datetime.datetime.now().strftime("%I:%M%p %b %d, %Y"),
            "settings": {
                "angle_means": self.angle_means
            }
        }
        for split in self.splits:
            if split not in complete_dict:
                complete_dict[split] = copy.deepcopy(EMPTY_SPLIT_DICT)
            for p in self.get_protein_list_by_split_name(split):
                p.numpy()
                complete_dict[split]["ang"].append(p.angles)
                complete_dict[split]["seq"].append(p.seq)
                complete_dict[split]["ids"].append(p.id)
                complete_dict[split]["evo"].append(p.evolutionary)
                complete_dict[split]["msk"].append(p.mask)
                if p.has_hydrogens:
                    p.hcoords = torch.tensor(p.hcoords)
                    p.coords = p.hydrogenrep_to_heavyatomrep().numpy()
                complete_dict[split]["crd"].append(p.coords)
                complete_dict[split]["sec"].append(p.secondary_structure)
                complete_dict[split]["res"].append(p.resolution)
                complete_dict[split]["ums"].append(p.unmodified_seq)
                complete_dict[split]["mod"].append(p.is_modified)

        if not description:
            description = "Pickled SCNDataset."
        complete_dict['description'] = description

        with open(path, "wb") as f:
            pickle.dump(complete_dict, f)

        return

    def get_pnids(self):
        """Return a list of all protein IDs in the dataset."""
        return sorted(list(self.ids_to_SCNProtein.keys()))

    def to_fasta(self, path, ids=None):
        """Save the dataset to a FASTA file with one line per protein."""
        if ids is None:
            ids = self.get_pnids()
        with open(path, "w") as f:
            for pid in ids:
                p = self[pid]
                f.write(f">{p.id}\n{p.seq}\n")

    def to_fastas(self, path, ids=None):
        """Save the dataset to a directory of FASTA files, one per protein."""
        if ids is None:
            ids = self.get_pnids()
        os.makedirs(path, exist_ok=True)
        for pid in tqdm.tqdm(ids):
            p = self[pid]
            with open(os.path.join(path, f"{p.id}.fasta"), "w") as f:
                f.write(f">{p.id}\n{p.seq}\n")


if __name__ == "__main__":
    import sidechainnet as scn
    d = scn.load("debug",
                 scn_dataset=True,
                 complete_structures_only=True,
                 trim_edges=True,
                 scn_dir="/home/jok120/sidechainnet_data")
    d.filter(lambda p: len(p) < 50)
