import numpy as np

from sidechainnet.dataloaders.collate import pad_for_batch
from sidechainnet.utils.sequence import VOCAB
from sidechainnet.structure.build_info import NUM_COORDS_PER_RES
import sidechainnet as scn


class BatchedStructureBuilder(object):

    # Assumes sequence of integers

    def __init__(self, seq_batch, ang_batch=None, crd_batch=None, return_as_list=True):
        # Validate input data
        if (ang_batch is None and crd_batch is None) or (ang_batch is not None and
                                                         crd_batch is not None):
            raise ValueError("You must provide exactly one of either coordinates (crd) "
                             "or angles (ang).")
        if len(seq_batch.shape) != 2:
            raise ValueError(
                "The batch of sequences must have shape (batch_size x sequence_length). "
                "One-hot vectors are not yet supported.")
        if ang_batch is not None:
            self.ang_or_crd_batch = ang_batch
            self.uses_coords = False
        else:
            self.ang_or_crd_batch = crd_batch
            self.uses_coords = True

        self.return_as_list = return_as_list
        self.structure_builders = []
        self.unbuildable_structures = []
        self.max_seq_len = 0
        for i, (seq, ang_or_crd) in enumerate(zip(seq_batch, self.ang_or_crd_batch)):
            seq, ang_or_crd = unpad_tensors(seq, ang_or_crd)
            try:
                self.structure_builders.append(scn.StructureBuilder(seq, ang_or_crd))
            except ValueError as e:
                if self.uses_coords:
                    raise e
                # This means that we attempted to create StructureBuilder objects using
                # incomplete angle tensors (this is undefined/unsupported).
                self.unbuildable_structures.append(i)
                self.structure_builders.append(None)

            if len(seq) > self.max_seq_len:
                self.max_seq_len = len(seq)

    def build(self):
        all_coords = []
        for sb in self.structure_builders:
            all_coords.append(sb.build())
        if self.return_as_list:
            return all_coords
        else:
            return pad_for_batch(all_coords, self.max_seq_len, dtype='crd')

    def to_3Dmol(self, idx, style=None, **kwargs):
        """Generate protein structure & return interactive py3Dmol.view for visualization.

        Args:
            idx (int): index of the StructureBuilder to visualize.
            style (str, optional): Style string to be passed to py3Dmol for
                visualization. Defaults to None.

        Returns:
            py3Dmol.view object: A view object that is interactive in iPython notebook
                settings.
        """
        if not 0 <= idx < len(self.structure_builders):
            raise ValueError("provided index is not available.")
        if idx in self.unbuildable_structures:
            self._missing_residue_error(idx)
        return self.structure_builders[idx].to_3Dmol(style, **kwargs)

    def to_pdb(self, idx, path, title=None):
        """Generate protein structure & create a PDB file for specified protein.

        Args:
            idx (int): index of the StructureBuilder to visualize.
            path (str): Path to save PDB file.
            title (str, optional): Title of generated structure (default = 'pred').
        """
        if not 0 <= idx < len(self.structure_builders):
            raise ValueError("provided index is not available.")
        if idx in self.unbuildable_structures:
            self._missing_residue_error(idx)
        return self.structure_builders[idx].to_pdb(path, title)

    def to_gltf(self, idx, path, title=None):
        """Save protein structure as a GLTF (3D-object) file to given path.

        Args:
            idx (int): index of the StructureBuilder to visualize.
            path (str): Path to save GLTF file.
            title (str, optional): Title of generated structure (default = 'pred').
        """
        if not 0 <= idx < len(self.structure_builders):
            raise ValueError("provided index is not available.")
        if idx in self.unbuildable_structures:
            self._missing_residue_error(idx)
        return self.structure_builders[idx].to_gltf(path, title)

    def _missing_residue_error(self, structure_idx):
        """Raises a ValueError describing missing residues."""
        missing_loc = np.where((self.ang_or_crd_batch[structure_idx] == 0).all(axis=-1))
        raise ValueError(f"Building atomic coordinates from angles is not supported "
                         f"for structures with missing residues. Missing residues = "
                         f"{list(missing_loc[0])}. Protein structures with missing "
                         "residues are only supported if built directly from "
                         "coordinates (also supported by StructureBuilder).")

    def __delitem__(self, key):
        raise NotImplementedError("Deletion is not supported.")

    def __getitem__(self, key):
        return self.structure_builders[key]

    def __setitem__(self, key, value):
        self.structure_builders[key] = value


def unpad_tensors(sequence, other):
    """Unpads both sequence and other tensor based on the batch padding in sequence.

    This relies on the fact that the sequence tensor ONLY has pad characters to represent
    batch-level padding, not missing residues. This pad character can be used to identify
    the correct batch padding.

    Args:
        sequence (torch.LongTensor): Sequence represented as integer tensor.
        other (torch.Tensor): Angle or coordinate data as a torch.FloatTensor.
    
    Returns:
        Sequence and other tensors with the batch-level padding removed.
    """
    batch_mask = sequence.ne(VOCAB.pad_id)
    sequence = sequence[batch_mask]

    if other.shape[-1] == 3:
        # Working with coordinates
        other = other[:sequence.shape[0] * NUM_COORDS_PER_RES]
    elif other.shape[-1] == 12:
        # Working with angles
        other = other[:sequence.shape[0]]
    else:
        raise ValueError(f"Unknown other shape, {other.shape}.")

    return sequence, other
