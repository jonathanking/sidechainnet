from sidechainnet.structure.build_info import NUM_COORDS_PER_RES
from sidechainnet.utils.download import MAX_SEQ_LEN

from sidechainnet.utils.sequence import VOCAB, DSSPVocabulary
import torch
import numpy as np


class ProteinBatch(object):
    """Represents a batch of Protein objects that can be collated, constructed, etc."""

    def __init__(self, proteins, batch_pad_char=0) -> None:
        self.proteins = proteins
        self.batch_pad_char = batch_pad_char
        self.max_len = max((len(p) for p in self.proteins))

    def __getitem__(self, idx):
        return self.proteins[idx]

    @property
    def angles(self):
        angles = [p.angles for p in self]
        return self.pad_for_batch(angles, dtype='ang')

    def _get_seqs(self, one_hot=True):
        seqs = [p.int_seq for p in self]  # We request int representation rather than str
        return self.pad_for_batch(seqs, dtype='seq', seqs_as_onehot=one_hot, vocab=VOCAB)

    @property
    def seqs_int(self):
        return self._get_seqs(one_hot=False)

    @property
    def seqs_onehot(self):
        return self._get_seqs(one_hot=True)

    @property
    def seqs(self):
        return self.seqs_onehot

    @property
    def secondary(self, one_hot=True):
        secs = [p.int_secondary for p in self]
        return self.pad_for_batch(secs,
                                  dtype='seq',
                                  seqs_as_onehot=one_hot,
                                  vocab=DSSPVocabulary())

    @property
    def masks(self):
        return self.pad_for_batch([p.int_mask for p in self], 'msk')

    @property
    def evolutionary(self):
        return self.pad_for_batch([p.evolutionary for p in self], 'pssm')

    @property
    def coords(self):
        return self.pad_for_batch([p.coords for p in self], 'crd')

    @property
    def is_modified(self):
        return self.pad_for_batch([p.is_modified for p in self], 'msk')

    @property
    def ids(self):
        return [p.id for p in self]

    @property
    def resolutions(self):
        return [p.resolution for p in self]

    def __iter__(self):
        for p in self.proteins:
            yield p

    def pad_for_batch(self, items, dtype="", seqs_as_onehot=False, vocab=None):
        """Pad a list of items to batch_length using values dependent on the item type.

        Args:
            items: List of items to pad (i.e. sequences or masks represented as arrays of
                numbers, angles, coordinates, pssms).
            dtype: A string ('seq', 'msk', 'pssm', 'ang', 'crd') reperesenting the type of
                data included in items.
            seqs_as_onehot: Boolean. If True, sequence-type data will be returned in 1-hot
                vector form.
            vocab: DSSPVocabulary or ProteinVocabulary. Vocabulary object for translating
                and handling sequence-type data.

        Returns:
            A padded list of the input items, all independently converted to Torch tensors.
        """
        # TODO: Check how pad chars are used in the embedding and model
        batch = []
        if dtype == "seq":
            # Sequences are padded with a specific VOCAB pad character
            for seq in items:
                z = np.ones((self.max_len - len(seq))) * vocab.pad_id
                c = np.concatenate((seq, z), axis=0)
                batch.append(c)
            batch = np.array(batch)
            batch = batch[:, :MAX_SEQ_LEN]
            batch = torch.LongTensor(batch)
            if seqs_as_onehot:
                batch = torch.nn.functional.one_hot(batch, len(vocab))
                if vocab.include_pad_char:
                    # Delete the column for the pad character since it is implied (0-vector)
                    if len(batch.shape) == 3:
                        batch = batch[:, :, :-1]
                    elif len(batch.shape) == 2:
                        batch = batch[:, :-1]
                    else:
                        raise ValueError(
                            f"Unexpected batch dimension {str(batch.shape)}.")
        elif dtype == "msk":
            # Mask sequences (1 if present, 0 if absent) are padded with 0s
            for msk in items:
                z = np.zeros((self.max_len - len(msk)))
                c = np.concatenate((msk, z), axis=0)
                batch.append(c)
            batch = np.array(batch)
            batch = batch[:, :MAX_SEQ_LEN]
            batch = torch.LongTensor(batch)
        elif dtype in ["pssm", "ang"]:
            # Mask other features with 0-vectors of a matching shape
            for item in items:
                z = np.zeros((self.max_len - len(item), item.shape[-1]))
                c = np.concatenate((item, z), axis=0)
                batch.append(c)
            batch = np.array(batch)
            batch = batch[:, :MAX_SEQ_LEN]
            batch = torch.FloatTensor(batch)
        elif dtype == "crd":
            for item in items:
                z = np.zeros(
                    (self.max_len * NUM_COORDS_PER_RES - len(item), item.shape[-1]))
                c = np.concatenate((item, z), axis=0)
                batch.append(c)
            batch = np.array(batch)
            # There are multiple rows per res, so we allow the coord matrix to be larger
            batch = batch[:, :MAX_SEQ_LEN * NUM_COORDS_PER_RES]
            batch = torch.FloatTensor(batch)

        return batch
    
    def __len__(self):
        return len(self.proteins)
    
    def __str__(self):
        return f"ProteinBatch(n={len(self)}, max_len={self.max_len})"