""" Utility functions when generating protein structure. """

import re

import numpy as np
import prody as pr

from protein_transformer.protein.SidechainBuildInfo import SC_BUILD_INFO
from protein_transformer.protein.structure_exceptions import \
    NonStandardAminoAcidError, IncompleteStructureError, SequenceError, \
    ContigMultipleMatchingError, ShortStructureError, MissingAtomsError, \
    NoneStructureError
from protein_transformer.protein.Structure import NUM_PREDICTED_ANGLES, \
    NUM_BB_TORSION_ANGLES, NUM_BB_OTHER_ANGLES, NUM_PREDICTED_COORDS
from protein_transformer.protein.Sequence import AA_MAP, AA_MAP_INV

GLOBAL_PAD_CHAR = np.nan

def get_backbone_from_full_coords(crds, invert=False):
    """
    #TODO Implement corresponding function for angles
    Given a coordinate tensor that may or may not have a batch dimension,
    this function returns the same coordinate tensor but excludes all the
    sidechain coordinates.
    """
    mask = np.array([1, 1, 1] + [0] * (NUM_PREDICTED_COORDS - 3), dtype=np.bool)
    if invert:
        mask = np.invert(mask)
    if len(crds.shape) == 2:
        return crds[np.tile(mask, crds.shape[0]//NUM_PREDICTED_COORDS), :]
    else:
        return crds[:,np.tile(mask, crds.shape[1]//NUM_PREDICTED_COORDS), :]


def get_sidechain_from_full_coords(crds):
    """
    Given a coordinate tensor that may or may not have a batch dimension,
    this function returns the same coordinate tensor but excludes all the
    backbone coordinates.
    """
    return get_backbone_from_full_coords(crds, invert=True)


def parse_astral_summary_file(path):
    """
    Given a path to the ASTRAL database summary file, this function parses
    that file and returns a dictionary that maps ASTRAL IDs to (pdbid, chain).
    """
    d = {}
    for line in open(path, "r").readlines():
        if line.startswith("#"):
            continue
        line_items = line.split()
        if line_items[3] == "-":
            continue
        if line_items[3] not in d.keys():
            d[line_items[3]] = (line_items[4], line_items[5])
    return d


def get_chain_from_astral_id(astral_id, d):
    """
    Given an ASTRAL ID and the ASTRAL->PDB/chain mapping dictionary,
    this function attempts to return the relevant, parsed ProDy object.
    """
    pdbid, chain = d[astral_id]
    assert "," not in chain, f"Issue parsing {astral_id} with chain {chain} and pdbid {pdbid}."
    chain, resnums = chain.split(":")
    a = pr.parsePDB(pdbid, chain=chain) # TODO maybe change this to CIF?
    if resnums != "":
        if resnums[0] == "-":
            # Ranges with negative numbers must be escaped with ` character
            a = a.select(f"resnum `{resnums[0] + resnums[1:].replace('-', ' to ')}`")
        else:
            a = a.select(f"resnum {resnums.replace('-', ' to ')}")
    return a


def get_header_seq_from_astral_id(astral_id, d):
    """
    Attempts to return the sequence associated with a given ASTRAL ID.
    Requires the ASTRAL->PDB/chain mapping dictionary.
    # TODO might not want to do this anymore, maybe using wrong file/IDs
    """
    pdbid, chain = d[astral_id]
    assert "," not in chain, f"Issue parsing {astral_id} with chain {chain} and pdbid {pdbid}."
    chain, resnums = chain.split(":")
    resnums = ''.join(i for i in resnums if (i.isdigit() or i == "-"))
    a, h = pr.parsePDB(pdbid, chain=chain, header=True)
    if resnums == "":
        return h[chain].sequence
    else:
        # This means the ASTRAL id is a substructure, and I can't use the seqres record directly
        raise SequenceError


def angle_list_to_sin_cos(angs, reshape=True):
    """
    Given a list of angles, returns a new list where those angles have been
    turned into their sines and cosines. If reshape is False, a new dim. is
    added that can hold the sine and cosine of each angle, i.e. (len x #angs)
    -> ( len x #angs x 2). If reshape is true, this last dim. is squashed so
    that the list of angles becomes [cos sin cos sin ...].
    """
    new_list = []
    for a in angs:
        new_mat = np.zeros((a.shape[0], a.shape[1], 2))
        new_mat[:, :, 0] = np.cos(a)
        new_mat[:, :, 1] = np.sin(a)
        if reshape:
            new_list.append(new_mat.reshape(-1, NUM_PREDICTED_ANGLES * 2))
        else:
            new_list.append(new_mat)
    return new_list


def seq_to_onehot(seq):
    """
    Given an AA sequence, returns a vector of one-hot vectors.
    """
    vector_array = []
    for aa in seq:
        one_hot = np.zeros(len(AA_MAP), dtype=bool)
        if aa in AA_MAP.keys():
            one_hot[AA_MAP[aa]] = 1
        else:
            one_hot -= 1
        vector_array.append(one_hot)
    return np.asarray(vector_array)


def onehot_to_seq(oh):
    """
    Given a vector of one-hot vectors, returns its corresponding AA sequence.
    """
    seq = ""
    for aa in oh:
        idx = aa.argmax()
        seq += AA_MAP_INV[idx]
    return seq


def check_standard_continuous(residue, prev_res_num):
    """
    Asserts that the residue is standard and that the chain is continuous.
    """
    if not residue.isstdaa:
        raise NonStandardAminoAcidError("Found a non-std AA.")
    if residue.getResnum() != prev_res_num:
        raise IncompleteStructureError("Chain is missing residues.")
    return True


def determine_sidechain_atomnames(_res):
    """
    Given a residue from ProDy, returns a list of sidechain atom names that
    must be recorded.
    """
    if _res.getResname() in SC_BUILD_INFO.keys():
        return SC_BUILD_INFO[_res.getResname()]["atom-names"]
    else:
        raise NonStandardAminoAcidError


def compute_sidechain_dihedrals(residue, prev_residue, next_res):
    """
    Computes all angles to predict for a given residue. If the residue is the
    first in the protein chain, a fictitious C atom is placed before the
    first N. This is used to compute a [ C-1, N, CA, CB] dihedral angle. If
    it is not the first residue in the chain, the previous residue's C is
    used instead. Then, each group of 4 atoms in atom_names is used to
    generate a list of dihedral angles for this residue.
    """
    try:
        torsion_names = SC_BUILD_INFO[residue.getResname()]["torsion-names"]
    except KeyError:
        raise NonStandardAminoAcidError
    if len(torsion_names) == 0:
        return (NUM_PREDICTED_ANGLES - (NUM_BB_TORSION_ANGLES + NUM_BB_OTHER_ANGLES)) * [GLOBAL_PAD_CHAR]

    # Compute CB dihedral, which may depend on the previous or next residue for placement
    try:
        if prev_residue:
            cb_dihedral = compute_single_dihedral((prev_residue.select("name C"),
                                       *(residue.select(f"name {an}") for an in ["N", "CA", "CB"])))
        else:
            cb_dihedral = compute_single_dihedral((next_res.select("name N"),
                                       *(residue.select(f"name {an}") for an in ["C", "CA", "CB"])))
    except AttributeError:
        cb_dihedral = GLOBAL_PAD_CHAR

    res_dihedrals = [cb_dihedral]

    for t_name, t_val in zip(torsion_names[1:], SC_BUILD_INFO[residue.getResname()]["torsion-vals"][1:]):
        # Only record torsional angles that are relevant (i.e. not planar).
        # All torsion values that vary are marked with 'p' in SC_BUILD_INFO
        if t_val != "p":
            break
        atom_names = t_name.split("-")
        res_dihedrals.append(compute_single_dihedral([residue.select("name " + an) for an in atom_names]))

    return res_dihedrals + (NUM_PREDICTED_ANGLES - (NUM_BB_TORSION_ANGLES + NUM_BB_OTHER_ANGLES) - len(res_dihedrals)) * [GLOBAL_PAD_CHAR]


def get_atom_coords_by_names(residue, atom_names):
    """
    Given a ProDy Residue and a list of atom names, this attempts to select
    and return all the atoms. If atoms are not present, it substitutes the
    pad character in lieu of their coordinates.
    """
    coords = []
    pad_coord = np.asarray([GLOBAL_PAD_CHAR]*3)
    for an in atom_names:
        a = residue.select(f"name {an}")
        if a:
            coords.append(a.getCoords()[0])
        else:
            coords.append(pad_coord)
    return coords


def measure_res_coordinates(_res):
    """
    Given a ProDy residue, measure all relevant coordinates.
    """
    sc_atom_names = determine_sidechain_atomnames(_res)
    bbcoords = get_atom_coords_by_names(_res, ["N", "CA", "C", "O"])
    sccoords = get_atom_coords_by_names(_res, sc_atom_names)
    coord_padding = np.zeros((NUM_PREDICTED_COORDS - len(bbcoords) - len(sccoords), 3))
    coord_padding[:] = GLOBAL_PAD_CHAR
    return np.concatenate((np.stack(bbcoords + sccoords), coord_padding))


def empty_coord():
    """
    Return an empty coordinate tensor, representing 1 padding character at
    the residue level.
    """
    coord_padding = np.zeros((NUM_PREDICTED_COORDS, 3))
    coord_padding[:] = GLOBAL_PAD_CHAR
    return coord_padding


def empty_ang():
    """
    Return an empty angle tensor, representing 1 padding character at the
    residue level.
    """
    dihe_padding = np.zeros(NUM_PREDICTED_ANGLES)
    dihe_padding[:] = GLOBAL_PAD_CHAR
    return dihe_padding


def find_contig_locations(contigs, true_seq):
    """
    Given a list of contigs, returns their positions within true_seq. Raises
    errors if the matching is ambiguous or if the observed sequence contains
    residues not found in the true seq.
    """
    contig_locs = []
    search_seq = str(true_seq)
    for i in range(len(contigs)):
        loc = search_seq.find(contigs[i])
        if len(re.findall(contigs[i], search_seq)) > 1:
            print(f"Multiple matches of {contigs[i]} found in {search_seq}.")
            raise ContigMultipleMatchingError
        if loc == -1:
            print(f"Can't find contig in search_seq.\n{contigs[i]}\n{search_seq}")
            raise SequenceError
        search_seq = search_seq[loc + len(contigs[i]):]
        if len(contig_locs) == 0:
            contig_locs.append(loc)
        else:
            contig_locs.append(contig_locs[-1] + len(contigs[i - 1]) + loc)
    return contig_locs


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


def use_contigs_to_compute_mask(contigs, true_seq, observed_sequence):
    """
    Given a list of contigs, aka contiguous sequence portions from a protein
    structure, along with the true sequence and an obs. sequence that may
    contain gaps, this function returns the mask and true sequence. The mask
    has '-' for gaps and '+' for present items w.r.t. the true sequence.
    """
    mask_seq = "-" * len(true_seq)
    # If there are no gaps in the structure, then keep the observed sequence
    if len(contigs) == 1:
        true_seq = observed_sequence
        mask_seq = "+" * len(observed_sequence)
    else:
        # Compute the best contig locations
        contig_locs = find_contig_locations(contigs, true_seq)

        # Use the contig locations to create our mask
        for contig, c_loc in zip(contigs, contig_locs):
            mask_seq = mask_seq[:c_loc] + "+" * len(contig) + mask_seq[c_loc + len(contig):]

        # Now that we have our mask sequence, we can trim its ends and fill in pad chars for residues
        mask_seq, true_seq = trim_mask_and_true_seqs(mask_seq, true_seq)
    return mask_seq, true_seq


def update_contigs(contigs, current_contig, all_residues, res_id):
    """
    This function updates the state variables `contigs` and `cur_contigs`
    while get_seq_and_masked_coords_and_angles processes the protein. These
    variables are later used to compute the missing residue masks for
    protein. This works by recording all of the continuous regions of the
    protein, and then attempting to identify what residues were missing
    inbetween those contiguous regions.
    """
    res = all_residues[res_id]
    if current_contig == "":
        current_contig = res.getSequence()[0]
    if res_id < len(all_residues) - 1 and residues_are_contiguous(res, all_residues[res_id + 1]):
        # The residues are connected
        current_contig += all_residues[res_id + 1].getSequence()[0]
    elif res_id < len(all_residues) - 1 and not residues_are_contiguous(res, all_residues[res_id + 1]):
        # The residues are not connected
        contigs.append(current_contig)
        current_contig = ""
    return contigs, current_contig

def get_seq_and_masked_coords_and_angles(chain, true_seq):
    """
    Given a ProDy Chain object (from a Hierarchical View), return a tuple
    (angles, coords, sequence). Returns None if the PDB should be ignored due
    to weird artifacts. Also measures the bond angles along the peptide
    backbone, since they account for significant variation.
    i.e. [[phi, psi, omega, ncac, cacn, cnca, chi1, chi2,...chi12],[...] ...]
    """
    chain = chain.select("protein and not hetero")
    if chain is None:
        raise NoneStructureError
    if chain.nonstdaa:
        raise NonStandardAminoAcidError
    chain = chain.copy()

    coords = []
    dihedrals = []
    observed_sequence = ""
    all_residues = list(chain.iterResidues())
    if len(all_residues) < 2:
        raise ShortStructureError
    prev_res = None
    next_res = all_residues[1]

    # Mask info
    current_contig = ""
    contigs = []  # TODO get rid of contigs, replace with absolutely true seq...Must find ABSOLUTELY TRUE SEQ

    for res_id, res in enumerate(all_residues):
        if not res.stdaa:
            raise NonStandardAminoAcidError
        # Measure basic angles
        bb_angles = measure_phi_psi_omega(res)
        bond_angles = measure_bond_angles(res, res_id, all_residues)

        # Measure sidechain angles
        all_res_angles = bb_angles + bond_angles + compute_sidechain_dihedrals(res, prev_res, next_res)

        # Measure coordinates
        rescoords = measure_res_coordinates(res)

        # Update records
        coords.append(rescoords)
        dihedrals.append(all_res_angles)
        prev_res = res
        observed_sequence += res.getSequence()[0]

        contigs, current_contig = update_contigs(contigs, current_contig, all_residues, res_id)

    if current_contig != "":
        contigs.append(current_contig)

    mask_seq, true_seq = use_contigs_to_compute_mask(contigs, true_seq, observed_sequence)

    assert mask_seq.count("+") == len(coords), f"The number of coords ({len(coords)}) must match the " \
        f"number of '+'s in mask_seq {mask_seq.count('+')}, {mask_seq}.\n{observed_sequence}\n{true_seq}"
    assert mask_seq.count("+") == len(dihedrals), f"The number of dihedrals ({len(dihedrals)}) must match the" \
        f" number of '+'s in mask_seq {mask_seq.count('+')}, {mask_seq}."

    # Use the mask to fill in missing residues
    coords, dihedrals = use_mask_to_pad_coords_dihedrals(mask_seq, coords, dihedrals)
    assert len(coords) == len(true_seq), "True sequence and coordinates must be same size at end of analysis"

    dihedrals_np = np.asarray(dihedrals)
    coords_np = np.concatenate(coords)

    if coords_np.shape[0] != len(true_seq) * NUM_PREDICTED_COORDS:
        print(f"Coords shape {coords_np.shape} does not match len(seq)*13 = "
              f"{len(true_seq) * NUM_PREDICTED_COORDS},\nOBS: {observed_sequence}\nTRU: {true_seq}\n{chain}")
        raise SequenceError
    # TODO return true sequence and mask here, should have same string length
    return dihedrals_np, coords_np, true_seq


def residues_are_contiguous(resA, resB):
    """
     Returns True if resA is connected to resB.
     """
    contiguous_threshold = 2  # The maximum allowed distance for a peptide bond
    try:
        cur_coords = resA.select("name C").getCoords()
        next_coords = resB.select("name N").getCoords()
    except AttributeError as e:
        # raise MissingBackboneAtomsError("Residue")
        return resA.getResnum() + 1 == resB.getResnum()
    return np.linalg.norm(cur_coords - next_coords) <= contiguous_threshold


def no_nans_infs_allzeros(matrix):
    """
    Returns true if a matrix does not contain NaNs, infs, or all 0s.
    """
    return not np.any(np.isinf(matrix)) and np.any(matrix)


def get_bond_angles(res, next_res):
    """
    Given 2 residues, returns the ncac, cacn, and cnca bond angles between
    them. If any atoms are not present, the corresponding angles are set to
    the GLOBAL_PAD_CHAR. If next_res is None, then only NCAC is measured.
    """
    # First residue angles
    n1, ca1, c1 = tuple(res.select(f"name {a}") for a in ["N", "CA", "C"])

    if n1 and ca1 and c1:
        ncac = safecalcAngle(n1, ca1, c1, radian=True)
    else:
        ncac = GLOBAL_PAD_CHAR

    # Second residue angles
    if next_res is None:
        return ncac, GLOBAL_PAD_CHAR, GLOBAL_PAD_CHAR
    n2, ca2 = (next_res.select(f"name {a}") for a in ["N", "CA"])
    if ca1 and c1 and n2:
        cacn = safecalcAngle(ca1, c1, n2, radian=True)
    else:
        cacn = GLOBAL_PAD_CHAR
    if c1 and n2 and ca2:
        cnca = safecalcAngle(c1, n2, ca2, radian=True)
    else:
        cnca = GLOBAL_PAD_CHAR
    return ncac, cacn, cnca


def safecalcAngle(a, b, c, radian):
    """
    Calculates the angle between 3 coordinates. If any of them are missing, the
    function raises a MissingAtomsError.
    """
    try:
        angle = pr.calcAngle(a, b, c, radian=radian)[0]
    except ValueError:
        raise MissingAtomsError
    return angle



def measure_bond_angles(residue, res_idx, all_res):
    """
    Given a residue, measure the ncac, cacn, and cnca bond angles.
    """
    if res_idx == len(all_res) - 1:
        next_res = None
    else:
        next_res = all_res[res_idx + 1]
    return list(get_bond_angles(residue, next_res))


def measure_phi_psi_omega(residue, include_OXT=False):
    """
    Returns phi, psi, omega for a residue, replacing out-of-bounds angles
    with GLOBAL_PAD_CHAR.
    """
    try:
        phi = pr.calcPhi(residue, radian=True, dist=None)
    except ValueError:
        phi = GLOBAL_PAD_CHAR
    try:
        psi = pr.calcPsi(residue, radian=True, dist=None)
    except ValueError:
        # For the last residue, we can measure a "psi" angle that is actually
        # the placement of the terminal oxygen. Currently, this is not utilized
        # in the building of structures, but it is included in case the need
        # arises in the future. Otherwise, this would simply become a pad
        # character.
        if include_OXT:
            try:
                psi = compute_single_dihedral(residue.select("name N CA C OXT"))
            except ValueError:
                psi = GLOBAL_PAD_CHAR
            except IndexError:
                psi = GLOBAL_PAD_CHAR
        else:
            psi = GLOBAL_PAD_CHAR
    try:
        omega = pr.calcOmega(residue, radian=True, dist=None)
    except ValueError:
        omega = GLOBAL_PAD_CHAR
    return [phi, psi, omega]


def compute_single_dihedral(atoms):
    """
    Given an iterable of 4 Atoms, calculate the dihedral angle between them
    in radians.
    """
    if None in atoms:
        return GLOBAL_PAD_CHAR
    else:
        atoms = [a.getCoords()[0] for a in atoms]
        return get_dihedral(atoms[0], atoms[1], atoms[2], atoms[3], radian=True)


def get_dihedral(coords1, coords2, coords3, coords4, radian=False):
    """
    Returns the dihedral angle in degrees. Modified from
    prody.measure.measure to use a numerically safe normalization method.
    """
    rad2deg = 180 / np.pi
    eps = 1e-6

    a1 = coords2 - coords1
    a2 = coords3 - coords2
    a3 = coords4 - coords3

    v1 = np.cross(a1, a2)
    v1 = v1 / (v1 * v1).sum(-1) ** 0.5
    v2 = np.cross(a2, a3)
    v2 = v2 / (v2 * v2).sum(-1) ** 0.5
    porm = np.sign((v1 * a3).sum(-1))
    arccos_input_raw = (v1 * v2).sum(-1) / ((v1 ** 2).sum(-1) * (v2 ** 2).sum(-1)) ** 0.5
    if -1 <= arccos_input_raw <= 1:
        arccos_input = arccos_input_raw
    elif arccos_input_raw > 1 and arccos_input_raw - 1 < eps:
        arccos_input = 1
    elif arccos_input_raw < -1 and np.abs(arccos_input_raw) - 1 < eps:
        arccos_input = -1
    else:
        raise ArithmeticError("Numerical issue with input to arccos.")
    rad = np.arccos(arccos_input)
    if not porm == 0:
        rad = rad * porm
    if radian:
        return rad
    else:
        return rad * rad2deg
