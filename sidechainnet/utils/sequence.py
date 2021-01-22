"""Implement utilities for handling protein sequences and vocabularies."""

import numpy as np

from sidechainnet.structure.build_info import NUM_COORDS_PER_RES, NUM_ANGLES
from sidechainnet.utils.measure import GLOBAL_PAD_CHAR


def trim_mask_and_true_seqs(mask_seq, true_seq):
    """Given an equal-length mask and sequence, removes gaps from the ends of both."""
    mask_seq_no_left = mask_seq.lstrip('-')
    mask_seq_no_right = mask_seq.rstrip('-')
    n_removed_left = len(mask_seq) - len(mask_seq_no_left)
    n_removed_right = len(mask_seq) - len(mask_seq_no_right)
    n_removed_right = None if n_removed_right == 0 else -n_removed_right
    true_seq = true_seq[n_removed_left:n_removed_right]
    mask_seq = mask_seq.strip("-")
    return mask_seq, true_seq


def empty_coord():
    """Return an empty coordinate tensor representing 1 residue-level pad character."""
    coord_padding = np.zeros((NUM_COORDS_PER_RES, 3))
    coord_padding[:] = GLOBAL_PAD_CHAR
    return coord_padding


def empty_ang():
    """Return an empty angle tensor representing 1 residue-level pad character."""
    dihe_padding = np.zeros(NUM_ANGLES)
    dihe_padding[:] = GLOBAL_PAD_CHAR
    return dihe_padding


def use_mask_to_pad_coords_dihedrals(mask_seq, coords, dihedrals):
    """Given a mask sequence ('-' for gap, '+' for present), and python lists of
    coordinates and dihedrals, this function places gaps in the relevant locations for
    each before returning.

    At the end, both should have the same length as the mask_seq.
    """
    new_coords = []
    new_angs = []
    coords = iter(coords)
    dihedrals = iter(dihedrals)
    for m in mask_seq:
        if m == "+":
            new_coords.append(next(coords))
            new_angs.append(next(dihedrals))
        else:
            new_coords.append(empty_coord())
            new_angs.append(empty_ang())
    return new_coords, new_angs


def bin_sequence_data(seqs, maxlen):
    """Given a list of sequences and a maximum training length, this function bins the
    sequences by their lengths (using numpy's 'auto' parameter), and then records the
    histogram information, as well as some statistics. This information is returned as a
    dictionary.

    This function allows the user to avoid computing this information at the
    start of each training run.
    """
    lens = list(map(lambda x: len(x) if len(x) <= maxlen else maxlen, seqs))
    hist_counts, hist_bins = np.histogram(lens, bins="auto")
    hist_bins = hist_bins[
        1:]  # make each bin define the rightmost value in each bin, ie '( , ]'.
    bin_probs = hist_counts / hist_counts.sum()
    bin_map = {}

    # Compute a mapping from bin number to index in dataset
    seq_i = 0
    bin_j = 0
    while seq_i < len(seqs):
        if lens[seq_i] <= hist_bins[bin_j]:
            try:
                bin_map[bin_j].append(seq_i)
            except KeyError:
                bin_map[bin_j] = [seq_i]
            seq_i += 1
        else:
            bin_j += 1

    return {
        "hist_counts": hist_counts,
        "hist_bins": hist_bins,
        "bin_probs": bin_probs,
        "bin_map": bin_map,
        "bin_max_len": maxlen
    }


class ProteinVocabulary(object):
    """Represents the 'vocabulary' of amino acids for encoding a protein sequence.

    Includes pad, sos, eos, and unknown characters as well as the 20 standard
    amino acids.
    """

    def __init__(self,
                 add_sos_eos=False,
                 include_unknown_char=False,
                 include_pad_char=True):
        self.include_unknown_char = include_unknown_char
        self.include_pad_char = include_pad_char
        self.pad_char = "_"  # Pad character
        self.unk_char = "?"  # unknown character
        self.sos_char = "<"  # SOS character
        self.eos_char = ">"  # EOS character

        self._char2int = dict()
        self._int2char = dict()

        # Extract the ordered list of 1-letter amino acid codes from the project-level
        # AA_MAP.
        self.stdaas = map(lambda x: x[0], sorted(list(AA_MAP.items()),
                                                 key=lambda x: x[1]))
        self.stdaas = "".join(filter(lambda x: len(x) == 1, self.stdaas))
        for aa in self.stdaas:
            self.add(aa)

        if include_pad_char:
            self.add(self.pad_char)
            self.pad_id = self[self.pad_char]
        else:
            self.pad_id = 0  # Implicit padding with all-zeros
        if include_unknown_char:
            self.add(self.unk_char)
        if add_sos_eos:
            self.add(self.sos_char)
            self.add(self.eos_char)
            self.sos_id = self[self.sos_char]
            self.eos_id = self[self.eos_char]

    def __getitem__(self, aa):
        if self.include_unknown_char:
            return self._char2int.get(aa, self._char2int[self.unk_char])
        else:
            return self._char2int.get(aa, self._char2int[self.pad_char])

    def __contains__(self, aa):
        return aa in self._char2int

    def __setitem__(self, key, value):
        raise ValueError('vocabulary is readonly')

    def __len__(self):
        return len(self._char2int)

    def __repr__(self):
        return f"ProteinVocabulary[size={len(self)}]"

    def int2char(self, idx):
        return self._int2char[idx]

    def int2chars(self, idx):
        return ONE_TO_THREE_LETTER_MAP[self._int2char[idx]]

    def add(self, aa):
        if aa not in self:
            aaid = self._char2int[aa] = len(self)
            self._int2char[aaid] = aa
            return aaid
        else:
            return self[aa]

    def str2ints(self, seq, add_sos_eos=True):
        if add_sos_eos:
            return [self["<"]] + [self[aa] for aa in seq] + [self[">"]]
        else:
            return [self[aa] for aa in seq]

    def ints2str(self, ints, include_sos_eos=False, exclude_pad=False):
        seq = ""
        for i in ints:
            c = self.int2char(i)
            if exclude_pad and c == self.pad_char:
                continue
            if include_sos_eos or (c not in [self.sos_char, self.eos_char, self.pad_char
                                            ]):
                seq += c
        return seq


class DSSPVocabulary(object):
    """Represents the 'vocabulary' of DSSP secondary structure codes."""

    def __init__(self, add_sos_eos=False):
        self.include_pad_char = True
        self.pad_char = " "  # Pad character
        self.sos_char = "<"  # SOS character
        self.eos_char = ">"  # EOS character

        codes = DSSP_CODES + " "

        if add_sos_eos:
            codes += "<>"
        self._char2int = {c: i for (i, c) in enumerate(codes)}
        self._int2char = {v: k for (k, v) in self._char2int.items()}
        self.pad_id = self._char2int[" "]

    def __getitem__(self, aa):
        if self.include_unknown_char:
            return self._char2int.get(aa, self._char2int[self.unk_char])
        else:
            return self._char2int.get(aa, self._char2int[self.pad_char])

    def __contains__(self, aa):
        return aa in self._char2int

    def __len__(self):
        return len(self._char2int)

    def __repr__(self):
        return f"DSSPVocabulary[size={len(self)}]"

    def int2char(self, idx):
        return self._int2char[idx]

    def str2ints(self, seq, add_sos_eos=True):
        if add_sos_eos:
            return [self._char2int["<"]] + [self._char2int[c] for c in seq
                                           ] + [self._char2int[">"]]
        else:
            return [self._char2int[c] for c in seq]


DSSP_CODES = dssp_codes = "BEGHILST"

ONE_TO_THREE_LETTER_MAP = {
    "R": "ARG",
    "H": "HIS",
    "K": "LYS",
    "D": "ASP",
    "E": "GLU",
    "S": "SER",
    "T": "THR",
    "N": "ASN",
    "Q": "GLN",
    "C": "CYS",
    "G": "GLY",
    "P": "PRO",
    "A": "ALA",
    "V": "VAL",
    "I": "ILE",
    "L": "LEU",
    "M": "MET",
    "F": "PHE",
    "Y": "TYR",
    "W": "TRP"
}
THREE_TO_ONE_LETTER_MAP = {v: k for k, v in ONE_TO_THREE_LETTER_MAP.items()}

AA_MAP = {
    'A': 0,
    'C': 1,
    'D': 2,
    'E': 3,
    'F': 4,
    'G': 5,
    'H': 6,
    'I': 7,
    'K': 8,
    'L': 9,
    'M': 10,
    'N': 11,
    'P': 12,
    'Q': 13,
    'R': 14,
    'S': 15,
    'T': 16,
    'V': 17,
    'W': 18,
    'Y': 19
}
AA_MAP_INV = {v: k for k, v in AA_MAP.items()}

for one_letter_code in list(AA_MAP.keys()):
    AA_MAP[ONE_TO_THREE_LETTER_MAP[one_letter_code]] = AA_MAP[one_letter_code]

# TODO: create VOCAB object only when needed.
VOCAB = ProteinVocabulary()
