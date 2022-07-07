"""Utilities for building structures using sublinear algorithms."""

import numpy as np
import torch
from sidechainnet.structure.fastbuild_matrices import MakeBackbone, MakeTMats
from sidechainnet.structure.HydrogenBuilder import NUM_COORDS_PER_RES_W_HYDROGENS
from .build_info import (ANGLE_NAME_TO_IDX_MAP, BB_BUILD_INFO, NUM_COORDS_PER_RES,
                         SC_ANGLES_START_POS, SC_HBUILD_INFO)

#################################################################################
# Data structures for describing how to build all the atoms of each residue
# For each datum, we have shape: residue_alphabet x max_num_sc_atoms == (60 x ncoords)
#################################################################################

AA = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T',
    'V', 'W', 'Y'
]
AA3to1 = {
    'ALA': 'A',
    'ARG': 'R',
    'ASN': 'N',
    'ASP': 'D',
    'CYS': 'C',
    'GLN': 'Q',
    'GLU': 'E',
    'GLY': 'G',
    'HIS': 'H',
    'ILE': 'I',
    'LEU': 'L',
    'LYS': 'K',
    'MET': 'M',
    'PHE': 'F',
    'PRO': 'P',
    'SER': 'S',
    'THR': 'T',
    'TRP': 'W',
    'TYR': 'Y',
    'VAL': 'V'
}
AA1to3 = {v: k for (k, v) in AA3to1.items()}
NUM2AA = {i: a for i, a in enumerate(AA)}
AA2NUM = {a: i for i, a in enumerate(AA)}


def _make_sc_heavy_atom_tensors():
    """Create a dict mapping build_info (lens, angs, types) to vals per residue identity.

        For every sidechain atom, we need to know:
            1) what atom it is building off of (<NUM_COORDS_PER_RES=14),
            2) the bond length,
            3) the theta bond angle,
            4) the chi torsion angle, and
            5) whether or not the torsion angle is defined as being offset from another
                angle (as in the case of branching atoms).

        For atoms where the torsion angles are inputs, we must specify angle source index.
        Index should should start from zero as it is assumed that only relevant angles
        will be passed in. Zero is start of sidechain atoms/angles.

    Returns:
        A dictionary mapping data names (the data nessary to compute NeRF and extend
        sidechains) to the data itself.
    """
    nres = len(AA)
    ncoords = NUM_COORDS_PER_RES - 4
    sc_source_atom = torch.zeros(nres, ncoords, dtype=torch.long)
    sc_bond_length = torch.zeros(nres, ncoords)
    sc_ctheta = torch.zeros(nres, ncoords)
    sc_stheta = torch.zeros(nres, ncoords)
    sc_cchi = torch.zeros(nres, ncoords)
    sc_schi = torch.zeros(nres, ncoords)
    sc_offset = torch.zeros(nres, ncoords)

    # 0: no atom, 1: regular torsion, 2: offset torsion, 3: constant
    sc_type = torch.zeros(nres, ncoords, dtype=torch.long)

    for a in range(nres):
        A = AA[a]
        a3 = AA1to3[A]
        info = SC_HBUILD_INFO[a3]
        for i in range(NUM_COORDS_PER_RES - 4):
            if i < len(info['torsion-names']) and not info['torsion-names'][i].split("-")[-1].startswith("H"):
                sc_bond_length[a][i] = info['bonds-vals'][i]
                sc_ctheta[a][i] = np.cos(np.pi - info['angles-vals'][i])
                sc_stheta[a][i] = np.sin(np.pi - info['angles-vals'][i])

                t = info['torsion-vals'][i]
                # Declare type of torsion angle
                # Type 0: No atom and no angle (no change from sc_type value of 0)
                # Type 1: Normal, predicted/measured torsion angle
                if t == 'p':
                    sc_type[a][i] = 1
                # Type 2: Inferred torsion angle (offset from previous angle by pi)
                elif t == 'i':
                    sc_type[a][i] = 2
                    sc_offset[a][i] = np.pi
                # Type 3: Constant angle, transformed by 2pi - t.
                else:
                    sc_type[a][i] = 3
                    sc_cchi[a][i] = np.cos(2 * np.pi - t)
                    sc_schi[a][i] = np.sin(2 * np.pi - t)

                src = info['torsion-names'][i].split('-')[-2]
                if src != 'CA':
                    sc_source_atom[a][i] = _get_atom_idx_in_torsion_list(src, info['torsion-names'])
                else:
                    sc_source_atom[a][i] = -1
    return {
        'bond_lengths': sc_bond_length,  # Each of these is a matrix containing relevant
        'cthetas': sc_ctheta,  # build info for all atoms in all residues.
        'sthetas': sc_stheta,
        'cchis': sc_cchi,  # For this reason, each is a (20 x 10) matrix,
        'schis': sc_schi,  # Where 20 is the number of different residues
        'types': sc_type,  # we may be building, and 10 is the maximum
        'offsets': sc_offset,  # number of atoms we may extend from the CA.
        'sources': sc_source_atom  # e.g. Tryptophan has 10 sidechain atoms.
    }


def _get_atom_idx_in_torsion_list(atom_name, tor_name_list):
    if atom_name == 'C':
        return -3
    elif atom_name == 'N':
        return -2
    for position, t_name in enumerate(tor_name_list):
        if t_name.split('-')[-1].strip() == atom_name.strip():
            return position
    raise ValueError(f"Could not find {atom_name} in {tor_name_list}.")


def _make_sc_all_atom_tensors():
    """Create a dict to map all-atom build_info (lens/angs/etc) to vals per res identity.

        For every sidechain atom, we need to know:
            1) what atom it is building off of (<NUM_COORDS_PER_RES=14),
            2) the bond length,
            3) the theta bond angle,
            4) the chi torsion angle, and
            5) whether or not the torsion angle is defined as being offset from another
                angle (as in the case of branching atoms).

        This is true for heavy atoms as well as hydrogens. First, we build the heavy atom
        arrays as in the function above. Then we append hydrogen information.

    Returns:
        A dictionary mapping data names (the data nessary to compute NeRF and extend
        sidechains) to the data itself.
    """
    nres = len(AA)
    ncoords = NUM_COORDS_PER_RES_W_HYDROGENS - 6  # TODO Make a new var to describe this
    sc_source_atom = torch.zeros(nres, ncoords, dtype=torch.long)
    sc_bond_length = torch.zeros(nres, ncoords)
    sc_ctheta = torch.zeros(nres, ncoords)
    sc_stheta = torch.zeros(nres, ncoords)
    sc_cchi = torch.zeros(nres, ncoords)
    sc_schi = torch.zeros(nres, ncoords)
    sc_offset = torch.zeros(nres, ncoords)
    sc_atom_name = [[] for _ in range(nres)]  # Just for notekeeping
    sc_chi = torch.zeros(nres, ncoords)

    # 0: no atom, 1: regular torsion, 2: offset torsion, 3: constant
    sc_type = torch.zeros(nres, ncoords, dtype=torch.long)

    for a in range(nres):
        A = AA[a]
        a3 = AA1to3[A]
        info = SC_HBUILD_INFO[a3]
        # First we add array data cooresponding to heavy atoms defined in SCBUILDINFO[RES]
        i = 0
        for i in range(ncoords):
            if i < len(info['torsion-vals']):
                an = info['torsion-names'][i].split('-')[-1].strip()
                sc_atom_name[a].append(an)
                sc_bond_length[a][i] = info['bonds-vals'][i]
                sc_ctheta[a][i] = np.cos(np.pi - info['angles-vals'][i])
                sc_stheta[a][i] = np.sin(np.pi - info['angles-vals'][i])

                src = info['torsion-names'][i].split('-')[-2].strip()
                if src != 'CA':
                    sc_source_atom[a][i] = _get_atom_idx_in_torsion_list(src, info['torsion-names'])
                else:
                    sc_source_atom[a][i] = -1

                t = info['torsion-vals'][i]
                # Declare type of torsion angle
                # Type 0: No atom and no angle (no change from sc_type value of 0)
                # Type 1: Normal, predicted/measured torsion angle
                if t == 'p':
                    sc_type[a][i] = 1
                # Type 2: Inferred torsion angle (offset from previous angle by pi)
                elif t == 'i':
                    sc_type[a][i] = 2
                    sc_offset[a][i] = np.pi
                # Type 3: Constant float-valued angle, transformed by 2pi - t.
                elif not isinstance(t, tuple):
                    sc_type[a][i] = 3
                    sc_cchi[a][i] = np.cos(2 * np.pi - t)
                    sc_schi[a][i] = np.sin(2 * np.pi - t)
                    sc_chi[a][i] = 2 * np.pi - t

                # The following types describe hydrogens using the tuple format:
                # ('hi', RELATM, OFFSETVAL)

                elif isinstance(t, tuple):
                    # This is a hydrogen placed relative to an earlier angle/atom
                    # Both the relative atom and the atom that needs plcmnt have same src
                    _, rel_atom, offset = t
                    rel_atom_idx = _get_atom_idx_in_torsion_list(rel_atom, info['torsion-names'])
                    atom_src = sc_source_atom[a][rel_atom_idx]
                    assert atom_src == sc_source_atom[a][i], f"{atom_src}, {sc_source_atom[a][i]}, {t}"
                    # if atom_src == -1:
                    #     # The source is CA, this means this is HA and should be rel to CB
                    #     assert an == 'HA' and rel_atom == 'CB', "Atom must be HA and relative to CB."
                    if info['torsion-vals'][rel_atom_idx] == 'p':
                        # Type 4: Inferred H angle dependent on predicted angle
                        # cchi and schi must be filled in later once pred angle is known
                        sc_type[a][i] = 4
                        sc_offset[a][i] = offset
                    elif (info['torsion-vals'][rel_atom_idx] != 'p' and
                          isinstance(info['torsion-vals'][rel_atom_idx], float)):
                        # Type 5:  Inferred H angle dependent on constant angle
                        # This means relative angle is constant and can be computed now
                        sc_type[a][i] = 5
                        src_atom_idx = sc_source_atom[a][i]
                        sc_offset[a][i] = offset
                        sc_cchi[a][i] = np.cos(sc_chi[a][src_atom_idx] + offset)
                        sc_schi[a][i] = np.sin(sc_chi[a][src_atom_idx] + offset)

                else:
                    raise ValueError('Unknown type for torsion value. ' +
                                     str(info['torsion-vals'][sc_source_atom[a][i]]))

            else:
                break

    return {
        'bond_lengths': sc_bond_length,  # Each of these is a matrix containing relevant
        'cthetas': sc_ctheta,  # build info for all atoms in all residues.
        'sthetas': sc_stheta,
        'cchis': sc_cchi,  # For this reason, each is a (20 x 10) matrix,
        'schis': sc_schi,  # Where 20 is the number of different residues
        'types': sc_type,  # we may be building, and 10 is the maximum
        'offsets': sc_offset,  # number of atoms we may extend from the CA.
        'sources': sc_source_atom,  # e.g. Tryptophan has 10 sidechain atoms.
        'names': sc_atom_name  # list of lists of atom names per residue (convenience)
    }


def _xNRES(v):  # make 20 copies as tensor
    return torch.Tensor([v]).repeat(len(AA)).reshape(len(AA), 1)


# In the dictionaries below, we specify geometries for atoms building off of N, CA, and C.
# Only CA has angles that vary by residue identity. The other atoms (N, C) have identical
# build data for every residue. CA atom build info is organized into a data with data
# labels as keys, and vectors of values ordered by residue identity.

SC_HEAVY_ATOM_BUILD_INFO = {
    # Nothing extends from Nitrogen, so no data is needed here.
    'N': None,
    # Many atoms potentially extend from alpha Carbons. Data is organized by res identity.
    # i.e. 'bond_lengths' : 20 x 10,  'types' : 20 x 10
    # Items are meant to be selected from this dictionary to match the sequence that will
    # be built.
    'CA': _make_sc_heavy_atom_tensors(),
    'HA': None,
    # Carbonyl carbons are built off of the backbone C. Each residue builds this oxygen
    # in the same way, so we simply repeat the relevant build info data per res identity.
    'C': {
        'bond_lengths': _xNRES(BB_BUILD_INFO["BONDLENS"]["c-o"]),
        'cthetas': _xNRES(np.cos(np.pi - BB_BUILD_INFO["BONDANGS"]["ca-c-o"])),
        'sthetas': _xNRES(np.sin(np.pi - BB_BUILD_INFO["BONDANGS"]["ca-c-o"])),
        'cchis': _xNRES(0.),
        'schis': _xNRES(0.),
        'types': _xNRES(1),
        'offsets': _xNRES(0.),
        'sources': _xNRES(-1),
        # 'names': [['O'] for _ in range(20)]
    }
}

# This dict is nearly identical to the one above but adds information for H building.

SC_ALL_ATOM_BUILD_INFO = {
    'N': {
        'bond_lengths': _xNRES(1.01),
        'cthetas': _xNRES(np.cos(np.deg2rad(60))),
        'sthetas': _xNRES(np.sin(np.deg2rad(60))),
        'cchis': _xNRES(0.),
        'schis': _xNRES(0.),
        'types': _xNRES(1),
        'offsets': _xNRES(0.),
        'sources': _xNRES(-1),
    },
    'CA': _make_sc_all_atom_tensors(),
    # =O off of C
    'C': {
        'bond_lengths': _xNRES(BB_BUILD_INFO["BONDLENS"]["c-o"]),
        'cthetas': _xNRES(np.cos(np.pi - BB_BUILD_INFO["BONDANGS"]["ca-c-o"])),
        'sthetas': _xNRES(np.sin(np.pi - BB_BUILD_INFO["BONDANGS"]["ca-c-o"])),
        'cchis': _xNRES(0.),
        'schis': _xNRES(0.),
        'types': _xNRES(1),
        'offsets': _xNRES(0.),
        'sources': _xNRES(-1),
    }
}
# Because proline does not have a hydrogen on CA, we set the build type for that atom to 0
# SC_ALL_ATOM_BUILD_INFO['HA']['types'][AA2NUM['P']] = 0
# Because Glycine has two hydrogens on CA that are slightly different than other res

###################################################
# Coordinates
###################################################


def make_sidechain_coords(backbone_mats, seq_aa_index, ang, build_info, bb_angs=None):
    """Create sidechain coordinates from backbone matrices given relevant auxilliary data.

    Our goal is to generate arrays for each datum (bond length, thetas, chis) necessary
    for building the next atom for every residue in the sequence.
    Once we have the data arrays, we convert these into transformation matrices.
    Finally, the transformation matrices can be applied to generate coordinates.
    """
    dtype = ang.dtype
    device = ang.device
    L = len(seq_aa_index)

    # We take the sin/cos vals of the angs for simplicity (they are used in making TMats)
    sins = torch.sin(ang)
    coss = torch.cos(ang)

    # Select out necessary build info for all atoms (up to 10) in all residues (up to L)
    bond_lengths = build_info['bond_lengths'][seq_aa_index].to(device).type(dtype)  # Lx10
    cthetas = build_info['cthetas'][seq_aa_index].to(device).type(dtype)
    sthetas = build_info['sthetas'][seq_aa_index].to(device).type(dtype)
    cchis = build_info['cchis'][seq_aa_index].to(device).type(dtype)
    schis = build_info['schis'][seq_aa_index].to(device).type(dtype)
    offsets = build_info['offsets'][seq_aa_index].to(device).type(dtype)
    types = build_info['types'][seq_aa_index].to(device)
    sources = build_info['sources'][seq_aa_index].to(device)
    names = [build_info['names'][idx] for idx in seq_aa_index] if 'names' in build_info else None

    # TODO modify the first and last residues (index 0 and -1 in above arrays) to add
    # TODO support for CB generation w/fictitious residue as well as terminal hydrogens.

    # Create empty sidechain coordinate tensor and starting coordinate at origin
    MAX = bond_lengths.shape[1]  # always equal to the maximum num sidechain atoms per res
    sccoords = torch.full((L, MAX, 3), torch.nan, device=device, dtype=dtype)
    vec = torch.tensor([0.0, 0, 0, 1], device=device, dtype=dtype)

    matrices = []
    masks = []

    # On our 1st iteration, we build CB from CA and must initialize 'previous' values.
    # We 1st identify the residues in the seq that are building an atom on this level i=0.
    # True if type of 0th atom is not 0 (aka an atom is being built, not true for Glycine)
    prevbuildmask = types[:, 0].bool()
    # We then select BB transformation matrices for the residues that will build an atom
    prevmats = backbone_mats[prevbuildmask]
    matrices.append(prevmats)

    # Now, iterate through each level of the sidechain, from 0 to the max number of atoms
    for i in range(MAX):
        # Select atom names to be built (convenience)
        if names is not None:
            anames = [anlist[i] if i < len(anlist) else None for anlist in names]

        # Identify all of the residues in the sequence that must build an atom at level i
        buildmask = types[:, i].bool()

        # Produce all of the bond lengths for the atoms to be built at level i
        lens = bond_lengths[buildmask, i]

        # If we have no atoms to build at level i, we are done.
        N = len(lens)
        if N == 0:
            break

        # Produce thetas, (non-torsional bond angs along sidechain not measured/predicted)
        ctheta = cthetas[buildmask, i]
        stheta = sthetas[buildmask, i]

        # Initialize chis, (torsional bond angs, set to predetermined constant, Types 3&5)
        cchi = cchis[buildmask, i]
        schi = schis[buildmask, i]

        # Fill in Type 1 chis (measured/predicted to build next atom, most common)
        type1mask = types[:, i] == 1
        if type1mask.any():
            # Of the chi angles for level i that are Type 1 (type1mask),
            # and of the residues that are building an atom at level i (buildmask),
            # Change those chi vals from constant to the vals that are measured/predicted.

            # Note that since type1mask is 'raw' and cchi is buildmask-ed, we apply
            # buildmask to type1mask before updating cchi.
            cchi[type1mask[buildmask]] = coss[type1mask, i]
            schi[type1mask[buildmask]] = sins[type1mask, i]

        # Fill in Type 2 chis (defined by an offset, i.e. branches)
        # type2mask = torch.logical_or(types[:, i] == 2, types[:, i] == 4)
        type2mask = types[:, i] == 2
        if type2mask.any():
            # re-used dihedral should always be positioned right after source position
            # Update the angle to be its offset plus the angle @ source pos + 1
            offang = ang[type2mask, sources[type2mask, i] + 1] + offsets[type2mask, i]
            cchi[type2mask[buildmask]] = torch.cos(offang)
            schi[type2mask[buildmask]] = torch.sin(offang)

        # Fill in Type 4 chis, (Hydrogens defined by an offset from a predicted atom)
        type4mask = types[:, i] == 4
        if type4mask.any():
            # re-used dihedral should always be positioned right after source position
            # Update the angle to be its offset plus the angle @ source pos + 1
            offang = ang[type4mask, sources[type4mask, i] + 1] + offsets[type4mask, i]
            cchi[type4mask[buildmask]] = torch.cos(offang)
            schi[type4mask[buildmask]] = torch.sin(offang)

        # Construct all necessary transformation matrices to build all atoms on level i
        tmats_for_level_i = MakeTMats.apply(ctheta, stheta, cchi, schi, lens)

        # Check if any atoms need a TMat that is from level lower (LL) than i - 1
        if (sources[:, i] != i - 1).any():
            prevmats = prevmats.clone()
            for k in range(i - 2, -2, -1):
                LLmask = (sources[:, i] == k) & buildmask
                if LLmask.any():
                    # change prevmat to earlier matrix
                    # We select the matrix from k+1 because when k==-1,
                    # we want the first matrix in the list
                    prevmats[LLmask[prevbuildmask]] = matrices[k+1][LLmask[masks[k]]]

        # We can finally update the globally-referenced TMats through multiplication
        prevmats = prevmats[buildmask[prevbuildmask]] @ tmats_for_level_i

        # Then produce the corresponding coordinates given all the global TMats
        c = prevmats @ vec
        sccoords[buildmask, i] = c[:, :3]

        # Record the global TMats and buildmasks for the next round of extension
        matrices.append(prevmats)
        masks.append(buildmask)
        prevbuildmask = buildmask

    return sccoords


def make_coords(seq, angles, add_hydrogens=False):
    """Create coordinates from sequence and angles using torch operations.

    build_info describes what atoms to make and how.
    """
    build_info = SC_HEAVY_ATOM_BUILD_INFO if not add_hydrogens else SC_ALL_ATOM_BUILD_INFO
    L = len(seq)
    device = angles.device
    dtype = angles.dtype

    ang_name_map = ANGLE_NAME_TO_IDX_MAP
    angles_cp = angles.clone()  # TODO still needed?

    # Construct an angle tensor populated with relevant backbone (BB) angles;
    # First: create a zero-filled angle tensor with shape (L+1 x NUM_BB_ANGS)
    ang = torch.zeros((L + 1, SC_ANGLES_START_POS), device=device, dtype=dtype)
    # Next: fill in the backbone angles, with pos 0 still zero
    ang[1:] = angles[:L, :SC_ANGLES_START_POS]
    # Because the first backbone angle is undefined (it has no phi), we replace nan with 0
    # This avoids improper generation of backbone TMats
    ang[1, 0] = 0
    ang[:, :3] = 2 * np.pi - ang[:, :3]  # backbone torsion angles, aka 'chi' for NeRF
    ang[:, 3:] = np.pi - ang[:, 3:]  # backbone bond angles, aka 'theta' for NeRF

    # Construct sin/cos representations of BB angle tensors
    sins = torch.sin(ang)
    coss = torch.cos(ang)

    # Construct flattenned tensors describing chi and theta angles for BB NeRF (L*3,).
    schi = sins[:, :3].flatten()[1:-2]
    cchi = coss[:, :3].flatten()[1:-2]
    stheta = sins[:, 3:].flatten()[1:-2]
    ctheta = coss[:, 3:].flatten()[1:-2]

    # Construct flattenned tensor describing lens between BB atoms (L*3,) for all N-CA-C.
    lens = torch.tensor([
        BB_BUILD_INFO['BONDLENS']['c-n'], BB_BUILD_INFO['BONDLENS']['n-ca'],
        BB_BUILD_INFO['BONDLENS']['ca-c']
    ],
                        device=device).repeat(L)
    lens[0] = 0  # first atom starts at origin and has no previous C atom len definition

    # Construct relative transformation matrices (TMats) per BB atom
    ncacM = MakeTMats.apply(ctheta, stheta, cchi, schi, lens)
    # Construct global TMats per BB atom by applying local TMats
    ncacM = MakeBackbone.apply(ncacM)

    # Initialize origin coordinate vector onto which TMats will be applied
    vec = torch.tensor([0.0, 0, 0, 1], device=device, dtype=dtype)
    # Construct N-CA-C BB coordinates by multiplying BB TMats and origin coordinate vec
    ncac = (ncacM @ vec)[:, :3].reshape(L, 3, 3)

    # Convert sequence to fixed AA index
    seq_aa_index = torch.tensor([AA2NUM[a] for a in seq], device=device)

    # Construct carbonyl oxygens (C=O) along backbone;
    # First: Select relevant angle (np.pi + psi)
    oang = (np.pi + ang[1:, ang_name_map['psi']]).unsqueeze(-1)
    # Next: Use the oang angle tensor and C-atom relevant info to place all oxygens
    ocoords = make_sidechain_coords(ncacM[2::3], seq_aa_index, oang, build_info['C'])

    # Create hydrogens for nitrogens
    if build_info['N']:  # nH
        nang = (np.pi + ang[:L, ang_name_map['omega']]).unsqueeze(-1)
        ncoords = make_sidechain_coords(ncacM[0::3], seq_aa_index, nang, build_info['N'])
    else:
        ncoords = torch.zeros(L, 0, 3, dtype=dtype, device=device)

    # Construct all sidechain atoms by iteratively applying TMats starting from CA
    scang = (2 * np.pi - angles_cp.to(device)[:, SC_ANGLES_START_POS:])
    sccoords = make_sidechain_coords(ncacM[1::3], seq_aa_index, scang, build_info['CA'],
                                     bb_angs=ang[1:].clone())

    # When present, we place the H and HAs atom (attached to N and CA) after the backbone
    # This allows the tail end of all residue representations to contain PADs only.
    return torch.concat([ncac, ocoords, ncoords, sccoords], dim=1)
