"""Utilities for building structures using sublinear algorithms."""

from .build_info import SC_BUILD_INFO, BB_BUILD_INFO, NUM_COORDS_PER_RES, SC_ANGLES_START_POS
import numpy as np
import torch

# setup tensor based residue information
aa = np.array([
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T',
    'V', 'W', 'Y'
])
aa3to1 = {
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
aa1to3 = {v: k for (k, v) in aa3to1.items()}
num2aa = {i: a for i, a in enumerate(aa)}
aa2num = {a: i for i, a in enumerate(aa)}


# For every sidechain atom, need to know:
#   1) what atom it is building off of (<NUM_COORDS_PER_RES=14),
#   2) the bond length,
#   3) the theta bond angle, and
#   4) the chi torsion angle.
# For atoms where the torsion angles are inputs, need to specify angle source index.
# Index should should start from zero as it is assumed that only relevant angles will be
# passed in. Zero is start of sidechain atoms/angles.

def _make_sc_ha_tensors():
    NC = NUM_COORDS_PER_RES - 4
    sc_source_atom = torch.zeros(20, NC, dtype=torch.long)
    sc_bond_length = torch.zeros(20, NC)
    sc_ctheta = torch.zeros(20, NC)
    sc_stheta = torch.zeros(20, NC)
    sc_cchi = torch.zeros(20, NC)
    sc_schi = torch.zeros(20, NC)
    sc_offset = torch.zeros(20, NC)

    # 0: no atom, 1: regular torsion, 2: offset torsion, 3: constant
    sc_type = torch.zeros(20, NC, dtype=torch.long)
    # chi idx is same as atom idx, unless offset torsion in which case it is one before

    for a in range(20):
        A = aa[a]
        a3 = aa1to3[A]
        info = SC_BUILD_INFO[a3]
        for i in range(NUM_COORDS_PER_RES - 4):
            if i < len(info['torsion-vals']):
                sc_bond_length[a][i] = info['bonds-vals'][i]
                sc_ctheta[a][i] = np.cos(np.pi - info['angles-vals'][i])
                sc_stheta[a][i] = np.sin(np.pi - info['angles-vals'][i])

                t = info['torsion-vals'][i]
                if t == 'p':
                    sc_type[a][i] = 1
                elif t == 'i':
                    sc_type[a][i] = 2
                    sc_offset[a][i] = np.pi
                else:
                    sc_type[a][i] = 3
                    sc_cchi[a][i] = np.cos(2 * np.pi - t)
                    sc_schi[a][i] = np.sin(2 * np.pi - t)

                src = info['bonds-names'][i].split('-')[0]
                if src != 'CA':
                    sc_source_atom[a][i] = info['atom-names'].index(src)
                else:
                    sc_source_atom[a][i] = -1
    return {
        'bond_lengths': sc_bond_length,  # Each of these is a matrix containing relevant
        'cthetas': sc_ctheta,            # build info for all atoms in all residues.
        'sthetas': sc_stheta,
        'cchis': sc_cchi,                # For this reason, each is a (20 x 10) matrix,
        'schis': sc_schi,                # Where 20 is the number of different residues
        'types': sc_type,                # we may be building, and 10 is the maximum
        'offsets': sc_offset,            # number of atoms we may extend from the CA.
        'sources': sc_source_atom        # e.g. Tryptophan has 10 sidechain atoms.
    }


# we specify geometries for atoms building off of N, CA, and C, but only
# CA has variable angles;  these are heavy atom only, but can be easily
# extended to hydrogen


def _x20(v):  # make 20 copies as tensor
    return torch.Tensor([v]).repeat(20).reshape(20, 1)


sc_heavy_atom_build_info = {
    'N': None,
    'CA': _make_sc_ha_tensors(),
    # =O off of C
    'C': {
        'bond_lengths': _x20(BB_BUILD_INFO["BONDLENS"]["c-o"]),
        'cthetas': _x20(np.cos(np.pi - BB_BUILD_INFO["BONDANGS"]["ca-c-o"])),
        'sthetas': _x20(np.sin(np.pi - BB_BUILD_INFO["BONDANGS"]["ca-c-o"])),
        'cchis': _x20(0.),
        'schis': _x20(0.),
        'types': _x20(1),
        'offsets': _x20(0.),
        'sources': _x20(-1)
    }
}

sc_all_atom_build_info = {
    'N': {
        'bond_lengths': _x20(1.01),
        'cthetas': _x20(np.cos(np.deg2rad(-60))),
        'sthetas': _x20(np.sin(np.deg2rad(-60))),
        'cchis': _x20(0.),
        'schis': _x20(0.),
        'types': _x20(1),
        'offsets': _x20(0.),
        'sources': _x20(-1)
    },
    'CA': _make_sc_ha_tensors(),  # TODO TODO TODO - add hydrogens
    # =O off of C
    'C': {
        'bond_lengths': _x20(BB_BUILD_INFO["BONDLENS"]["c-o"]),
        'cthetas': _x20(np.cos(np.pi - BB_BUILD_INFO["BONDANGS"]["ca-c-o"])),
        'sthetas': _x20(np.sin(np.pi - BB_BUILD_INFO["BONDANGS"]["ca-c-o"])),
        'cchis': _x20(0.),
        'schis': _x20(0.),
        'types': _x20(1),
        'offsets': _x20(0.),
        'sources': _x20(-1)
    }
}

###################################################
# Transformation Matrices Creation
###################################################


def makeTmats(ctheta, stheta, cchi, schi, lens):
    """Make transformation matrices given angles/lens.

    ctheta is used to determine device and number of matrices.
    """
    N = ctheta.shape[0]
    mats = torch.zeros((N, 4, 4), device=ctheta.device, dtype=ctheta.dtype)

    mats[:, 0, 0] = ctheta
    mats[:, 0, 1] = -stheta
    mats[:, 0, 3] = ctheta * lens

    mats[:, 1, 0] = cchi * stheta
    mats[:, 1, 1] = ctheta * cchi
    mats[:, 1, 2] = schi
    mats[:, 1, 3] = cchi * lens * stheta

    mats[:, 2, 0] = -stheta * schi
    mats[:, 2, 1] = -ctheta * schi
    mats[:, 2, 2] = cchi
    mats[:, 2, 3] = -lens * stheta * schi

    mats[:, 3, 3] = 1.0

    return mats


def makeTmats_backward(grad_outputs, ctheta, stheta, cchi, schi, lens):
    """Backwards function for makeTmats."""
    # yapf: disable
    grad = grad_outputs
    grad_ctheta = grad[:,0,0] + lens*grad[:,0,3] + cchi*grad[:,1,1] + -schi*grad[:,2,1]
    grad_stheta = -grad[:,0,1] + cchi*grad[:,1,0] + cchi*lens*grad[:,1,3] - schi*grad[:,2,0] - lens*schi*grad[:,2,3]
    grad_cchi = stheta*grad[:,1,0] + ctheta*grad[:,1,1] + lens*stheta*grad[:,1,3] + grad[:,2,2]
    grad_schi = grad[:,1,2] + -stheta*grad[:,2,0] - ctheta*grad[:,2,1] - lens*stheta*grad[:,2,3]
    # lengths are typically constant.. should we check requires grad?
    grad_lens = ctheta*grad[:,0,3] + cchi*stheta*grad[:,1,3] - stheta*schi*grad[:,2,3]
    # yapf: enable

    return grad_ctheta, grad_stheta, grad_cchi, grad_schi, grad_lens


class MakeTMats(torch.autograd.Function):

    @staticmethod
    def forward(ctx, ctheta, stheta, cchi, schi, lens):
        ctx.save_for_backward(ctheta, stheta, cchi, schi, lens)
        return makeTmats(ctheta, stheta, cchi, schi, lens)

    @staticmethod
    def backward(ctx, grad):
        return makeTmats_backward(grad, *ctx.saved_tensors)


###################################################
# Backbone
###################################################


def make_backbone_serial(ncacM):
    """Make backbone matrices from indiv. transformation matrices by multiplying in order.

    Modifies in-place.
    """
    for i in range(1, len(ncacM)):
        ncacM[i] = ncacM[i - 1] @ ncacM[i]
    return ncacM


def make_backbone_(M):
    """Make backbone matrices from indiv. transformation matrices using log(N) multiplies.

    Backbone matrices are created from inidividual transformation matrices via log(N)
    parallel matrix multiplies operations. Modifies in place.
    """
    N = len(M)
    indices = torch.arange(0, N, device=M.device)

    for i in range(N.bit_length()):
        dstmask = ((1 << i) & indices) != 0
        srci = ((-1 << i) & indices) - 1

        src = M[srci[dstmask]]
        dst = M[dstmask]

        M[dstmask] = src @ dst

    return M


def make_backbone_save(M):
    """Make backbone matrices from indiv. transformation matrices using log(N) multiplies.

    Make backbone matrices from individual transformation matrices using log(N) parallel
    matrix multiplies. Does not modify in place. Not very memory efficient in the
    interest of simplicity. Returns transformation matrices at each interation (last is
    final result).
    """
    N = len(M)
    bit_length = N.bit_length()

    indices = torch.arange(0, N, device=M.device)
    ret = torch.zeros(bit_length, N, 4, 4, device=M.device, dtype=M.dtype)

    for i in range(N.bit_length()):
        ret[i] = M
        dstmask = ((1 << i) & indices) != 0
        srci = ((-1 << i) & indices) - 1

        src = M[srci[dstmask]]
        dst = M[dstmask]

        M = M.clone()
        M[dstmask] = src @ dst

    return M, ret


def make_backbone_backward(grad, M, saved):
    """Compute gradient of M."""
    N = len(M)
    indices = torch.arange(0, N, device=grad.device)
    grad = grad.clone()  # make sure contiguous
    for i in range(N.bit_length() - 1, -1, -1):
        dstmask = ((1 << i) & indices) != 0
        srci = ((-1 << i) & indices) - 1
        srcmask = srci[dstmask]
        src = saved[i][srcmask]
        dst = saved[i][dstmask]

        srcgrad = grad[dstmask] @ dst.transpose(1, 2)  # src grad
        grad[dstmask] = src.transpose(1, 2) @ grad[dstmask]

        index = srcmask[:, None, None].repeat(1, 4, 4)
        grad.scatter_add_(0, index, srcgrad)

    return grad


class MakeBackbone(torch.autograd.Function):

    @staticmethod
    def forward(ctx, ncac):
        if ncac.requires_grad:
            M, saved = make_backbone_save(ncac)
            ctx.save_for_backward(ncac, saved)
            return M
        else:
            return make_backbone_(ncac.clone())

    @staticmethod
    def backward(ctx, grad):
        return make_backbone_backward(grad, *ctx.saved_tensors)


###################################################
# Coordinates
###################################################


def make_sidechain_coords(backbone, seq_aa_index, ang, build_info):
    """Create sidechain coordinates from backbone matrices given relevant auxilliary info.

    Given relevant backbone atom matrices, map from sequence to residue type, relevant
    variable angles, and  build info for chain of atoms off of residue,create coordinates.
    """
    dtype = ang.dtype
    device = ang.device
    L = len(seq_aa_index)

    sins = torch.sin(ang)
    coss = torch.cos(ang)

    # select out angles/bonds/etc for full seq

    bond_lengths = build_info['bond_lengths'][seq_aa_index].to(device).type(dtype)  # Lx10
    cthetas = build_info['cthetas'][seq_aa_index].to(device).type(dtype)
    sthetas = build_info['sthetas'][seq_aa_index].to(device).type(dtype)
    cchis = build_info['cchis'][seq_aa_index].to(device).type(dtype)
    schis = build_info['schis'][seq_aa_index].to(device).type(dtype)
    offsets = build_info['offsets'][seq_aa_index].to(device).type(dtype)
    types = build_info['types'][seq_aa_index].to(device)
    sources = build_info['sources'][seq_aa_index].to(device)

    MAX = bond_lengths.shape[1]
    sccoords = torch.full((L, MAX, 3), torch.nan, device=device, dtype=dtype)
    vec = torch.tensor([0.0, 0, 0, 1], device=device, dtype=dtype)

    matrices = []
    masks = []

    # first iteration: build from CA
    prevmask = types[:, 0].bool()  # no G
    prevmats = backbone[prevmask]

    for i in range(MAX):

        seqmask = types[:, i].bool()
        lens = bond_lengths[seqmask, i]
        N = len(lens)
        if N == 0:
            break
        # theta always constant
        ctheta = cthetas[seqmask, i]
        stheta = sthetas[seqmask, i]

        # set to constant
        cchi = cchis[seqmask, i]
        schi = schis[seqmask, i]

        # unless not
        regmask = types[:, i] == 1  # regular chi
        if regmask.any():
            cchi[regmask[seqmask]] = coss[regmask, i]
            schi[regmask[seqmask]] = sins[regmask, i]

        invmask = types[:, i] == 2
        if invmask.any():
            # re-used dihedral should always be positioned right after source position
            offang = ang[invmask, sources[invmask, i] + 1] + offsets[invmask, i]
            cchi[invmask[seqmask]] = torch.cos(offang)
            schi[invmask[seqmask]] = torch.sin(offang)

        # check if any atoms need matrix from farther back
        if (sources[:, i] != i - 1).any():
            prevmats = prevmats.clone()

            for k in range(i - 2, -1, -1):
                srcmask = (sources[:, i] == k) & seqmask
                if srcmask.any():  # change prevmat to earlier matrix
                    prevmats[srcmask[prevmask]] = matrices[k][srcmask[masks[k]]]

        mats = MakeTMats.apply(ctheta, stheta, cchi, schi, lens)

        prevmats = prevmats[seqmask[prevmask]] @ mats

        c = prevmats @ vec
        sccoords[seqmask, i] = c[:, :3]

        matrices.append(prevmats)
        masks.append(seqmask)
        prevmask = seqmask

    return sccoords


class MakeSCCoords(torch.autograd.Function):

    @staticmethod
    def forward(ctx, backbone, seq_aa_index, ang, build_info):
        ctx.save_for_backward(backbone, ang)
        ctx.build_info = build_info
        ctx.seq_aa_index = seq_aa_index
        return make_sidechain_coords(backbone, seq_aa_index, ang, build_info)

    @staticmethod
    def backward(ctx, grad):
        # implementing this can make make_coords 3X faster
        raise NotImplementedError


def make_coords(seq, angles, build_info=sc_heavy_atom_build_info):
    """Create coordinates from sequence and angles using torch operations.

    build_info describes what atoms to make and how.
    """
    L = len(seq)
    device = angles.device
    dtype = angles.dtype
    ang = torch.zeros((L + 1, SC_ANGLES_START_POS), device=device, dtype=dtype)
    ang[1:] = angles[:L, :SC_ANGLES_START_POS]
    ang[:, 3:] = np.pi - ang[:, 3:]      # theta
    ang[:, :3] = 2 * np.pi - ang[:, :3]  # chi

    sins = torch.sin(ang)
    coss = torch.cos(ang)
    schi = sins[:, :3].flatten()[1:-2]
    cchi = coss[:, :3].flatten()[1:-2]
    stheta = sins[:, 3:].flatten()[1:-2]
    ctheta = coss[:, 3:].flatten()[1:-2]

    lens = torch.tensor([
        BB_BUILD_INFO['BONDLENS']['c-n'], BB_BUILD_INFO['BONDLENS']['n-ca'],
        BB_BUILD_INFO['BONDLENS']['ca-c']
    ],
                        device=device).repeat(L)
    lens[0] = 0  # first atom starts at origin

    ncacM = MakeTMats.apply(ctheta, stheta, cchi, schi, lens)
    ncacM = MakeBackbone.apply(ncacM)

    # backbone coords
    vec = torch.tensor([0.0, 0, 0, 1], device=device, dtype=dtype)
    ncac = (ncacM @ vec)[:, :3].reshape(L, 3, 3)

    # convert sequence to aa index
    seq_aa_index = torch.tensor([aa2num[a] for a in seq], device=device)

    # =O
    oang = (np.pi + ang[1:, 1]).unsqueeze(-1)
    ocoords = make_sidechain_coords(ncacM[2::3], seq_aa_index, oang, build_info['C'])

    if build_info['N']:  # nH
        nang = (np.pi + ang[:L, 0]).unsqueeze(-1)
        ncoords = make_sidechain_coords(ncacM[0::3], seq_aa_index, nang, build_info['N'])
    else:
        ncoords = torch.zeros(L, 0, 3, dtype=dtype, device=device)

    # make sidechains off of CA
    scang = (2 * np.pi - angles.to(device)[:, SC_ANGLES_START_POS:])
    sccoords = make_sidechain_coords(ncacM[1::3], seq_aa_index, scang, build_info['CA'])

    return torch.concat([ncac, ocoords, sccoords, ncoords], dim=1)
