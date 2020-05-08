import numpy as np

from sidechainnet.utils.structure import empty_coord, empty_ang


def trim_mask_and_true_seqs(mask_seq, true_seq):
    """
    Given a mask and true sequence of the same length, this removes gaps from
    the ends of both.
    """
    mask_seq_no_left = mask_seq.lstrip('-')
    mask_seq_no_right = mask_seq.rstrip('-')
    n_removed_left = len(mask_seq) - len(mask_seq_no_left)
    n_removed_right = len(mask_seq) - len(mask_seq_no_right)
    n_removed_right = None if n_removed_right == 0 else -n_removed_right
    true_seq = true_seq[n_removed_left:n_removed_right]
    mask_seq = mask_seq.strip("-")
    return mask_seq, true_seq


def use_mask_to_pad_coords_dihedrals(mask_seq, coords, dihedrals):
    """
    Given a mask sequence ('-' for gap, '+' for present), and python lists of
    coordinates and dihedrals, this function places gaps in the relevant
    locations for each before returning. At the end, both should have the
    same length as the mask_seq.
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
    """
    Given a list of sequences and a maximum training length, this function
    bins the sequences by their lengths (using numpy's 'auto' parameter),
    and then records the histogram information, as well as some statistics.
    This information is returned as a dictionary.

    This function allows the user to avoid computing this information at the
    start of each training run.
    """
    lens = list(map(lambda x: len(x) if len(x) <= maxlen else maxlen, seqs))
    hist_counts, hist_bins = np.histogram(lens, bins="auto")
    hist_bins = hist_bins[1:]  # make each bin define the rightmost value in each bin, ie '( , ]'.
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

    return {"hist_counts": hist_counts,
            "hist_bins"  : hist_bins,
            "bin_probs"  : bin_probs,
            "bin_map"    : bin_map,
            "bin_max_len": maxlen}
