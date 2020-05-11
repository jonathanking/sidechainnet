"""
This file will implement functions that allow the merging of sidechain data with
Mohammed AlQuraishi's ProteinNet. It works by conforming the data I have
generated with sidechain information to match the sequence and mask reported by
ProteinNet.

Author: Jonathan King
Date : 3/09/2020
"""

import numpy as np
import torch
from Bio import Align
from tqdm import tqdm

from sidechainnet.utils.build_info import NUM_PREDICTED_COORDS


def init_aligner():
    """
    Creates an aligner whose weights penalize excessive gaps, make gaps
    in the ProteinNet sequence impossible, and prefer gaps at the tail ends
    of sequences.
    """
    a = Align.PairwiseAligner()

    # Don't allow for gaps or mismatches with the target sequence
    a.target_gap_score = -np.inf
    a.mismatch = -np.inf
    a.mismatch_score = -np.inf

    # Do not let matching items overwhelm determining where gaps should go
    a.match = 10

    # Generally, prefer to extend gaps than to create them
    a.query_extend_gap_score = 99
    a.query_open_gap_score = 49

    # Set slight preference for open gaps on the edges, however, if present, strongly prefer single edge gaps
    a.query_end_open_gap_score = 50
    a.query_end_extend_gap_score = 100

    return a


def get_mask_from_alignment(al):
    """ For a single alignment, return the mask as a string of '+' and '-'s. """
    alignment_str = str(al).split("\n")[1]
    return alignment_str.replace("|", "+")


def can_be_directly_merged(aligner, pn_seq, my_seq, pn_mask):
    """
    Returns True iff when pn_seq and my_seq are aligned, the resultant mask
    is the same as reported by ProteinNet. Also returns the computed_mask that
    matches with ProteinNet
    """
    a = aligner.align(pn_seq, my_seq)
    pn_mask = binary_mask_to_str(pn_mask)
    warning = None

    def masks_match(pn, new):
        return pn == new

    if len(a) == 0:
        warning = "failed"
        return False, None, None, warning

    elif len(a) == 1:
        a0 = a[0]
        computed_mask = get_mask_from_alignment(a0)
        if not masks_match(pn_mask, computed_mask):
            warning = "single alignment, mask mismatch"
        return True, computed_mask, a0, warning

    elif len(a) > 1:
        best_mask = None
        found_a_match = False
        best_alignment = None
        best_idx = 0
        if len(a) >= 50:
            many_alignments = True
            a = list(a)[:50]
        else:
            many_alignments = False
        for i, a0 in enumerate(a):
            computed_mask = get_mask_from_alignment(a0)
            if not best_mask:
                best_mask = computed_mask
                best_idx = i
            if not best_alignment:
                best_alignment = a0
            if masks_match(pn_mask, computed_mask):
                found_a_match = True
                best_mask = computed_mask
                best_alignment = a0
                best_idx = i
                break
        if found_a_match:
            warning = "multiple alignments, found matching mask"
            if many_alignments:
                warning += ", many alignments"
            return True, best_mask, best_alignment, warning
        else:
            mask = get_mask_from_alignment(a[0])
            warning = "multiple alignments, mask mismatch"
            if many_alignments:
                warning += ", many alignments"
            return True, mask, a[0], warning


def other_alignments_with_same_score(all_alignments, cur_alignment_idx,
                                     cur_alignment_score):
    """Returns True if there are other alignments with identical scores.

    Args:
        all_alignments: PairwiseAlignment iterable object from BioPython.Align
        cur_alignment_idx: The index of the desired alignment
        cur_alignment_score: The score of the desired alignment

    Returns:
        True if any alignments other than the one specified have scores that
        are identical to the specified alignment.
    """
    if len(all_alignments) <= 1:
        return False

    for i, a0 in enumerate(all_alignments):
        if i > 0 and a0.score < cur_alignment_score:
            break
        if i == cur_alignment_idx:
            continue
        elif a0.score == cur_alignment_score:
            return True

    return False


def binary_mask_to_str(m):
    """
    Given an iterable or list of 1s and 0s representing a mask, this returns
    a string mask with '+'s and '-'s.
    """
    m = list(map(lambda x: "-" if x == 0 else "+", m))
    return "".join(m)


def find_how_many_entries_can_be_directly_merged():
    """
    Counts the number of entries that can be successfully aligned between the
    sidechain dataset and the protein dataset.
    """
    d = torch.load(
        "/home/jok120/protein-transformer/data/proteinnet/casp12_200218_30.pt")
    pn = torch.load("/home/jok120/proteinnet/data/casp12/torch/training_30.pt")
    aligner = init_aligner()
    total = 0
    successful = 0
    with open("merging_problems.csv", "w") as f, open("merging_success.csv",
                                                      "w") as sf:
        for i, my_id in enumerate(tqdm(d["train"]["ids"])):
            my_seq, pn_seq, pn_mask = d["train"]["seq"][i], pn[my_id][
                "primary"], binary_mask_to_str(pn[my_id]["mask"])
            my_seq = unmask_seq(d["train"]["ang"][i], my_seq)
            result, computed_mask, alignment = can_be_directly_merged(
                aligner, pn_seq, my_seq, pn_mask)
            if result:
                successful += 1
                sf.write(",".join([my_id, my_seq, computed_mask]) + "\n")
            else:
                if pn_mask.count("+") < len(my_seq):
                    size_comparison = "<"
                elif pn_mask.count("+") > len(my_seq):
                    size_comparison = ">"
                else:
                    size_comparison = "=="
                f.write(
                    f"{my_id}: (PN {size_comparison} Obs)\n{str(alignment)}")
                f.write(f"PN Mask:\n{pn_mask}\n\n")
            total += 1
        print(
            f"{successful} out of {total} ({successful / total}) sequences can be merged successfully."
        )


def unmask_seq(ang, seq):
    """
    Given an angle array that is padded with np.nans, applies this padding to
    the sequence, and returns the sequence without any padding. This means
    that the input sequence contains residues that may be missing, while the
    returned sequence contains only observed residues.
    """
    mask = np.logical_not(np.isnan(ang).all(axis=-1))
    new_seq = ""
    for m, s in zip(mask, seq):
        if m:
            new_seq += s
        else:
            continue

    return new_seq


if __name__ == '__main__':
    find_how_many_entries_can_be_directly_merged()


def coordinate_iterator(coords, atoms_per_res):
    """Iterates over coordinates in a numpy array grouped by residue.

    Args:
        coords: Numpy array of coordinates. (L x atoms_per_res) x 3.
        atoms_per_res: Number of atomic coordinates per residue.

    Returns:
        An iterator where every next call returns the next atoms_per_res
            coordinates.
    """
    assert len(coords) % atoms_per_res == 0, f"There must be {atoms_per_res}" \
                                             f" atoms for every residue.\n" \
                                             f"len(coords) = {len(coords)}"
    i = 0
    while i + atoms_per_res <= len(coords):
        yield coords[i:i + atoms_per_res]
        i += atoms_per_res


def expand_data_with_mask(data, mask):
    """Uses mask to expand data as necessary.

    Args:
        data: May be evolutionary (numpy array, Lx21), secondary (unsupported),
            angles (2D numpy array, Lx12), coordinates (2D numpy array,
             (Lx13)x3).
        mask: String of '+'s and '-'s representing if data is present with
            respect to protein primary sequence.

    Returns:
        Data in the same format, possibly extending L to match the length of
        the mask, that now contains padding.
    """
    if mask.count("-") == 0:
        return data

    size = data.shape[-1]
    if size == 3:
        data = coordinate_iterator(data, NUM_PREDICTED_COORDS)
        blank = np.empty((NUM_PREDICTED_COORDS, 3))
    else:
        data = iter(data)
        blank = np.empty((size,))

    blank[:] = np.nan

    new_data = []
    for m in mask:
        if m == "+":
            new_data.append(next(data))
        elif m == "-":
            new_data.append(blank.copy())

    return np.vstack(new_data)


def pad_seq_with_mask(seq, mask):
    """Given a shorter sequence, expands it to match the padding in mask.

    Args:
        seq: String with length smaller than mask.
        mask: String of '+'s and '-'s used to expand seq.

    Returns:
        New string of seq but with added '-'s where indicated by mask.
    """
    seq_iter = iter(seq)
    new_seq = ""
    for m in mask:
        if m == "+":
            new_seq += next(seq_iter)
        elif m == "-":
            new_seq += "-"
    return new_seq