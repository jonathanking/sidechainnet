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
from sidechainnet.dataloaders.SCNProtein import SCNProtein


class SCNDataset(object):
    """A representation of a SidechainNet dataset."""

    def __init__(self, data) -> None:
        """Initialize a SCNDataset from underlying SidechainNet formatted dictionary."""
        super().__init__()
        # Determine available datasplits
        self.splits = []
        for split_name in ['train', 'valid', 'test']:
            for k in data.keys():
                if split_name in k:
                    self.splits.append(k)

        self.split_to_ids = {}
        self.ids_to_SCNProtein = {}
        self.idx_to_SCNProtein = {}

        # Create SCNProtein objects and add to data structure
        idx = 0
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
                self.ids_to_SCNProtein[i] = p
                self.idx_to_SCNProtein[idx] = p
                idx += 1

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
        yield from self.ids_to_SCNProtein.values()

    def __repr__(self) -> str:
        """Represent SCNDataset as a string."""
        return f"SCNDataset(n={len(self)})"

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
