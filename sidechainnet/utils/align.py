"""Functionality for aligning protein sequences in ProteinNet vs SidechainNet."""

import numpy as np
from Bio import Align

from sidechainnet.structure.build_info import NUM_COORDS_PER_RES, PRODY_CA_DIST
from sidechainnet.utils.download import ASTRAL_ID_MAPPING, determine_pnid_type
from sidechainnet.utils.measure import GLOBAL_PAD_CHAR


def init_basic_aligner(allow_mismatches=False):
    """Returns an aligner with minimal assumptions about gaps."""
    a = Align.PairwiseAligner()
    if allow_mismatches:
        a.mismatch_score = -1
        a.gap_score = -3
        a.target_gap_score = -np.inf
    if not allow_mismatches:
        a.mismatch = -np.inf
        a.mismatch_score = -np.inf
    return a


def init_aligner(allow_target_gaps=False, allow_target_mismatches=False):
    """Creates an aligner whose weights penalize excessive gaps, make gaps in the
    ProteinNet sequence impossible, and prefer gaps at the tail ends of sequences."""
    a = Align.PairwiseAligner()
    a.mismatch = -np.inf
    a.mismatch_score = -np.inf

    # Don't allow for gaps or mismatches with the target sequence
    if not allow_target_gaps:
        a.target_gap_score = -np.inf

    # Do not let matching items overwhelm determining where gaps should go
    if not allow_target_gaps:
        a.match = 10
    else:
        a.match = 200

    if allow_target_mismatches:
        a.mismatch = 200

    # Generally, prefer to extend gaps than to create them
    a.query_extend_gap_score = 99
    a.query_open_gap_score = 49

    # Set slight preference for open gaps on the edges, however, if present, strongly prefer single edge gaps
    a.query_end_open_gap_score = 50
    a.query_end_extend_gap_score = 100

    return a


def get_mask_from_alignment(al):
    """For a single alignment, return the mask as a string of '+' and '-'s."""
    alignment_str = str(al).split("\n")[1]
    return alignment_str.replace("|", "+")


def get_padded_second_seq_from_alignment(al):
    """For a single alignment, return the second padded string."""
    alignment_str = str(al).split("\n")[2]
    return alignment_str


def locate_char(c, s):
    """Returns a list of indices of character c in string s."""
    return [i for i, l in enumerate(s) if l == c]


def masks_match(pn, new):
    """Returns true if the two masks match, or if pn is a subset of new."""
    if pn == new:
        return True
    elif new.count("-") > pn.count("-"):
        # If all of the gaps specified by ProteinNet are found by our
        # alignment, but there are some additional gaps, this is acceptable.
        new_gap_locs = locate_char("-", new)
        pn_gap_locs = locate_char("-", pn)
        pn_gaps_still_present = all([pn_gap in new_gap_locs for pn_gap in pn_gap_locs])
        return pn_gaps_still_present
    else:
        return False


def shorten_ends(s1, s2, s1_ang, s1_crd, s1_raw_seq, s1_ismodified):
    """Shortens s1 by removing characters at either end that don't match s2.

    Args:
        s1: String, longer than s2
        s2: String

    Returns:
        A possibly shortened version of s1, with non-matching start and end
        characters trimmed off.
    """
    aligner = init_aligner(allow_target_gaps=True)
    a = aligner.align(s1, s2)
    mask = get_padded_second_seq_from_alignment(a[0])
    i = len(mask) - 1
    while mask[i] == "-":
        s1 = s1[:-1]
        s1_ang = s1_ang[:-1]
        s1_crd = s1_crd[:-NUM_COORDS_PER_RES]
        s1_raw_seq = s1_raw_seq[:-1]
        s1_ismodified = s1_ismodified[:-1]
        mask = mask[:-1]
        i -= 1
    while mask[0] == "-":
        s1 = s1[1:]
        s1_ang = s1_ang[1:]
        s1_crd = s1_crd[NUM_COORDS_PER_RES:]
        s1_raw_seq = s1_raw_seq[1:]
        s1_ismodified = s1_ismodified[1:]
        mask = mask[1:]
    return s1, s1_ang, s1_crd, s1_raw_seq, s1_ismodified


def merge(aligner, pn_entry, sc_entry, pnid, attempt_number=0, ignore_pnmask=False):
    """Returns True iff when pn_seq and my_seq are aligned, the resultant mask is the same
    as reported by ProteinNet.

    Also returns the computed_mask that matches with ProteinNet
    """
    pn_seq, pn_mask = pn_entry["primary"], pn_entry["mask"]
    my_seq, ang, crd, dssp = sc_entry["seq"], sc_entry["ang"], sc_entry["crd"], sc_entry[
        "sec"]
    unmod_seq = sc_entry['ums']
    is_modified = sc_entry['mod']

    a = aligner.align(pn_seq, my_seq)
    pn_mask = binary_mask_to_str(pn_mask)
    warning = None

    try:
        n_alignments = len(a)
    except OverflowError:
        n_alignments = 50
        warning = "failed"
        return None, None, ang, crd, dssp, unmod_seq, is_modified, warning

    if n_alignments == 0 and attempt_number == 0:
        # Use aligner with a typical set of assumptions.
        aligner = init_aligner()
        return merge(aligner,
                     pn_entry,
                     sc_entry,
                     pnid,
                     attempt_number=1,
                     ignore_pnmask=ignore_pnmask)

    if n_alignments == 0 and attempt_number == 1:
        # If there appear to be no alignments, it may be the case that there
        # were residues observed that were not present in the ProteinNet
        # sequence. If this occurs at the edges, we can safely trim the
        # observed sequence and try alignment once again
        my_seq, ang, crd, unmod_seq, is_modified = shorten_ends(
            my_seq, pn_seq, ang, crd, unmod_seq, is_modified)
        sc_entry['seq'] = my_seq
        sc_entry['ang'] = ang
        sc_entry['crd'] = crd
        sc_entry['ums'] = unmod_seq
        sc_entry['mod'] = is_modified
        return merge(aligner,
                     pn_entry,
                     sc_entry,
                     pnid,
                     attempt_number=2,
                     ignore_pnmask=ignore_pnmask)

    if n_alignments == 0 and attempt_number == 2:
        # Try making very few assumptions about gaps before allowing mismatches/gaps in
        # the target sequence.
        aligner = init_basic_aligner(allow_mismatches=True)
        return merge(aligner,
                     pn_entry,
                     sc_entry,
                     pnid,
                     attempt_number=3,
                     ignore_pnmask=ignore_pnmask)

    elif n_alignments == 0 and attempt_number == 3:
        aligner = init_aligner(allow_target_gaps=True, allow_target_mismatches=True)
        mask, a0, ang, crd, dssp, unmod_seq, is_modified, warning = merge(
            aligner,
            pn_entry,
            sc_entry,
            pnid,
            attempt_number=4,
            ignore_pnmask=ignore_pnmask)
        warning = warning + ", mismatch used in alignment" if warning else "mismatch used in alignment"
        return mask, a0, ang, crd, dssp, unmod_seq, is_modified, warning

    elif n_alignments == 0 and attempt_number == 4:
        warning = "failed"
        return None, None, ang, crd, dssp, unmod_seq, is_modified, warning

    elif n_alignments == 1:
        a0 = a[0]
        computed_mask = get_mask_from_alignment(a0)
        if attempt_number == 4:
            if computed_mask.count("X") + computed_mask.count(".") > 5:
                warning = "too many wrong AAs"
            computed_mask = computed_mask.replace("X", "+").replace(".", "+")
        if not masks_match(pn_mask, computed_mask) and not ignore_pnmask:
            if "astral" in determine_pnid_type(pnid):
                pdbid, chain = ASTRAL_ID_MAPPING[pnid.split("_")[1].replace("-", "_")]
                if "A" not in chain:
                    # This suggests that ProteinNet made a mistake and parsed
                    # chain A when they should have parsed the correct chain.
                    # This is therefore not an alignment error.
                    pass
                else:
                    # If the above case is not True, then we should still expect
                    # the mask we compute to match the one computed by ProteinNet
                    warning = "single alignment, mask mismatch"
            else:
                warning = "single alignment, mask mismatch"
        return computed_mask, a0, ang, crd, dssp, unmod_seq, is_modified, warning

    elif n_alignments > 1:
        best_mask = None
        found_a_match = False
        best_alignment = None
        best_idx = 0
        has_many_alignments = n_alignments >= 200
        for i, a0 in enumerate(a):
            if has_many_alignments and i >= 200:
                break
            computed_mask = get_mask_from_alignment(a0)
            if attempt_number == 4:
                if computed_mask.count("X") + computed_mask.count(".") > 5:
                    warning = "too many wrong AAs"
                computed_mask = computed_mask.replace("X", "+").replace(".", "+")
            if not best_mask:
                best_mask = computed_mask
                best_idx = i
            if not best_alignment:
                best_alignment = a0
            # if masks_match(pn_mask, computed_mask) or assert_mask_gaps_are_correct(
            #         computed_mask, crd)[0]:
            if assert_mask_gaps_are_correct(computed_mask, crd)[0]:
                found_a_match = True
                best_mask = computed_mask
                best_alignment = a0
                best_idx = i
                break
        if found_a_match:
            warning = "multiple alignments, found matching mask" if not warning else warning + ", multiple alignments, found matching mask"
            if has_many_alignments:
                warning += ", many alignments"
            return best_mask, best_alignment, ang, crd, dssp, unmod_seq, is_modified, warning
        else:
            mask = get_mask_from_alignment(a[0])
            warning = "multiple alignments, mask mismatch" if not warning else warning + ", multiple alignments, mask mismatch"
            if has_many_alignments:
                warning += ", many alignments"
            return mask, a[0], ang, crd, dssp, unmod_seq, is_modified, warning


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
    """Given an iterable or list of 1s and 0s representing a mask, this returns a string
    mask with '+'s and '-'s."""
    m = list(map(lambda x: "-" if x == 0 else "+", m))
    return "".join(m)


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
        Data in the same format, possibly extending L to match the length of,
        the mask, that now contains padding.
    """
    if (((isinstance(data, str) or isinstance(data, list)) and mask.count("-") == 0 and
         len(data) == len(mask)) or
        (not isinstance(data, str) and mask.count("-") == 0 and
         data.shape[0] == len(mask))):
        return data

    if isinstance(data, str):
        size = len(data)
        blank = " "
        data_iter = iter(data)
    elif isinstance(data, list):
        size = len(data)
        blank = "---"
        data_iter = iter(data)
    else:
        size = data.shape[-1] if len(data.shape) > 1 else 1
        if size == 3:
            data_iter = coordinate_iterator(data, NUM_COORDS_PER_RES)
            blank = np.empty((NUM_COORDS_PER_RES, 3))
            blank[:] = GLOBAL_PAD_CHAR
        elif size == 1:
            data_iter = iter(data)
            blank = 0
        else:
            data_iter = iter(data)
            blank = np.empty((size,))
            blank[:] = GLOBAL_PAD_CHAR

    new_data = []
    for m in mask:
        if m == "+" or m == ".":
            new_data.append(next(data_iter))
        elif m == "-" and (isinstance(data, str) or isinstance(data, list)):
            new_data.append(blank)
        elif m == "-" and size != 1:
            new_data.append(blank.copy())
        elif m == "-" and size == 1:
            new_data.append(blank)
        else:
            raise ValueError(f"Unknown mask character '{m}'.")

    if isinstance(data, str):
        return "".join(new_data)
    elif isinstance(data, list):
        return new_data
    elif size == 1:
        return np.asarray(new_data).astype("int8")  # Used for is_modified array of bits
    else:
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


def assert_mask_gaps_are_correct(mask, coordinates):
    """Returns True if the structure supports the mask.

    Args:
        mask: string of "+" and "-"s, denoting missing residues
        coordinates: numpy array (L x 14 x 3) of atomic coordinates

    Returns:
        True iff the mask is supported by the structure. If False, also returns length
        of the offending Ca-Ca distance.
    """
    CA_IDX = 1
    if mask.count("-") == 0:
        return True, 0

    # This should never happen
    if mask.count("+") != len(coordinates) // NUM_COORDS_PER_RES:
        return False, 0

    # First, build a nested list that holds all contiguous regions of the data
    # according to the mask
    coord_iter = coordinate_iterator(coordinates, NUM_COORDS_PER_RES)
    coord_contigs = []
    cur_contig = []

    for m in mask:
        if m == "-":
            if cur_contig != []:
                coord_contigs.append(cur_contig.copy())
                cur_contig = []
            continue
        else:
            cur_contig.append(next(coord_iter))
    if cur_contig != []:
        coord_contigs.append(cur_contig.copy())

    # Once the contiguous regions are reported, we check that the distance
    # between all alpha-carbons is less than ProDy's cutoff (4.1 Angstrom)
    resnum = 1
    for coord_contig in coord_contigs:
        if len(coord_contig) == 1:
            continue
        prev_ca = coord_contig[0][CA_IDX]
        for cur_res in coord_contig[1:]:
            cur_ca = cur_res[CA_IDX]
            if np.linalg.norm(cur_ca - prev_ca) > PRODY_CA_DIST * 1.85:
                return False, np.linalg.norm(cur_ca - prev_ca)
            prev_ca = cur_ca.copy()
            resnum += 1

    return True, 0
