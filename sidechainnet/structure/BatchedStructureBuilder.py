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
            ang_or_crd_batch = ang_batch
        else:
            ang_or_crd_batch = crd_batch

        self.return_as_list = return_as_list
        self.structure_builders = []
        self.max_seq_len = 0
        for seq, ang_or_crd in zip(seq_batch, ang_or_crd_batch):
            seq, ang_or_crd = unpad_tensors(seq, ang_or_crd)
            self.structure_builders.append(scn.StructureBuilder(seq, ang_or_crd))
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

