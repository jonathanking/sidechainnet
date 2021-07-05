"""Contains high-level functionality for protein structure building tools, i.e. NeRF."""

import numpy as np
import prody as pr
import torch
from sidechainnet.structure import StructureBuilder
from sidechainnet.structure.build_info import NUM_ANGLES
from sidechainnet.utils.measure import GLOBAL_PAD_CHAR
from sidechainnet.utils.sequence import VOCAB


def angles_to_coords(angles, seq, remove_batch_padding=False):
    """Convert torsional angles to coordinates."""
    pred_ang, input_seq = angles, seq
    if remove_batch_padding:
        # Remove batch-level masking
        batch_mask = input_seq.ne(VOCAB.pad_id)
        input_seq = input_seq[batch_mask]

    # Remove SOS and EOS characters if present
    # input_seq = remove_sos_eos_from_input(input_seq)
    # pred_ang = pred_ang[:input_seq.shape[0]]

    # Generate coordinates
    return generate_coords(pred_ang, input_seq, torch.device("cpu"))


def generate_coords(angles, input_seq, device):
    """Return a protein's coordinates generated from its angles and sequence.

    Given a tensor of angles (L x NUM_PREDICTED_ANGLES), produces the entire set
    of cartesian coordinates using the NeRF method, (L x A` x 3), where A` is
    the number of atoms generated (depends on amino acid sequence).
    """
    sb = StructureBuilder.StructureBuilder(input_seq, angles, device)
    return sb.build()


def nerf(a, b, c, l, theta, chi, l_bc=None, nerf_method="standard"):
    """Compute the position of point d given 3 previous points and several parameters.

    This function uses either sn_nerf or standard_nerf depending on the presence of l_bc.
    Both sn_nerf and standard_nerf follow the formulations in the paper by Jerod Parsons
    et al. https://doi.org/10.1002/jcc.20237 titled "Practical conversion from torsion
    space to Cartesian space for in silico protein synthesis."

    Args:
        a (torch.float32 tensor): (3 x 1) tensor describing point a.
        b (torch.float32 tensor): (3 x 1) tensor describing point b.
        c (torch.float32 tensor): (3 x 1) tensor describing point c.
        l_cd (torch.float32 tensor): (1) tensor describing the length between points
            c & d.
        theta (torch.float32 tensor): (1) tensor describing angle between points b, c,
            and d.
        chi (torch.float32 tensor): (1) tensor describing dihedral angle between points
            a, b, c, and d.
        l_bc (torch.float32 tensor, optional): (1) tensor describing length between points
            b and c.
        nerf_method (str, optional): Which NeRF implementation to use. "standard" uses
            the standard NeRF formulation described in many papers. "sn_nerf" uses an
            optimized version with less vector normalizations. Defaults to
            "standard".
    """
    if nerf_method == "standard" or l_bc is None:
        return standard_nerf(a, b, c, l, theta, chi)
    elif nerf_method == "sn_nerf" and l_bc is not None:
        return sn_nerf(a, b, c, l, theta, chi, l_bc)
    else:
        raise ValueError("l_bc must be provided for sn_nerf.")


def standard_nerf(a, b, c, l, theta, chi):
    """Compute the position of point d given 3 previous points and angle information.

    This implementation is based on the one originally written by Rohit Bhattacharya
    (rohit.bhattachar@gmail.com,
    https://github.com/rbhatta8/protein-design/blob/master/nerf.py). I have extended it to
    work with PyTorch. Original equation from https://doi.org/10.1002/jcc.20237.

    Args:
        a (torch.float32 tensor): (3 x 1) tensor describing point a.
        b (torch.float32 tensor): (3 x 1) tensor describing point b.
        c (torch.float32 tensor): (3 x 1) tensor describing point c.
        l_cd (torch.float32 tensor): (1) tensor describing the length between points
            c & d.
        theta (torch.float32 tensor): (1) tensor describing angle between points b, c,
            and d.
        chi (torch.float32 tensor): (1) tensor describing dihedral angle between points
            a, b, c, and d.

    Raises:
        ValueError: Raises ValueError when value of theta is not in [-pi, pi].

    Returns:
        torch.float32 tensor: (3 x 1) tensor describing coordinates of point c after
        placement using points a, b, c, and several parameters.
    """
    if not (-np.pi <= theta <= np.pi):
        raise ValueError(f"theta must be in radians and in [-pi, pi]. theta = {theta}")

    # calculate unit vectors AB and BC
    W_hat = torch.nn.functional.normalize(b - a, dim=0)
    x_hat = torch.nn.functional.normalize(c - b, dim=0)

    # calculate unit normals n = AB x BC
    # and p = n x BC
    n_unit = torch.cross(W_hat, x_hat)
    z_hat = torch.nn.functional.normalize(n_unit, dim=0)
    y_hat = torch.cross(z_hat, x_hat)

    # create rotation matrix [BC; p; n] (3x3)
    M = torch.stack([x_hat, y_hat, z_hat], dim=1)

    # calculate coord pre rotation matrix
    d = torch.stack([
        torch.squeeze(-l * torch.cos(theta)),
        torch.squeeze(l * torch.sin(theta) * torch.cos(chi)),
        torch.squeeze(l * torch.sin(theta) * torch.sin(chi))
    ])

    # calculate with rotation as our final output
    d = d.unsqueeze(1).to(torch.float32)
    res = c + torch.mm(M, d).squeeze()
    return res.squeeze()


def sn_nerf(a, b, c, l_cd, theta, chi, l_bc):
    """Return coordinates for point d given previous points & parameters. Optimized NeRF.

    This function has been optimized from the original nerf to be about 20% faster. It
    contains fewer normalization steps and total calculations than the original
    formulation. See https://doi.org/10.1002/jcc.20237 for details.

    Args:
        a (torch.float32 tensor): (3 x 1) tensor describing point a.
        b (torch.float32 tensor): (3 x 1) tensor describing point b.
        c (torch.float32 tensor): (3 x 1) tensor describing point c.
        l_cd (torch.float32 tensor): (1) tensor describing the length between points
            c & d.
        theta (torch.float32 tensor): (1) tensor describing angle between points b, c,
            and d.
        chi (torch.float32 tensor): (1) tensor describing dihedral angle between points
            a, b, c, and d.
        l_bc (torch.float32 tensor): (1) tensor describing length between points b and c.

    Raises:
        ValueError: Raises ValueError when value of theta is not in [-pi, pi].

    Returns:
        torch.float32 tensor: (3 x 1) tensor describing coordinates of point c after
        placement using points a, b, c, and several parameters.
    """
    if not (-np.pi <= theta <= np.pi):
        raise ValueError(f"theta must be in radians and in [-pi, pi]. theta = {theta}")
    AB = b - a
    BC = c - b
    bc = BC / l_bc
    n = torch.nn.functional.normalize(torch.cross(AB, bc), dim=0)
    n_x_bc = torch.cross(n, bc)

    M = torch.stack([bc, n_x_bc, n], dim=1)

    D2 = torch.stack([
        -l_cd * torch.cos(theta), l_cd * torch.sin(theta) * torch.cos(chi),
        l_cd * torch.sin(theta) * torch.sin(chi)
    ]).to(torch.float32).unsqueeze(1)

    D = torch.mm(M, D2).squeeze()

    return D + c


def determine_missing_positions(ang_or_coord_matrix):
    """Uses GLOBAL_PAD_CHAR to determine location of missing atoms or residues."""
    raise NotImplementedError


def deg2rad(angle):
    """Convert an angle in degrees to radians."""
    return angle * np.pi / 180.


def inverse_trig_transform(t):
    """Compute the atan2 of the last 2 dimensions of a given tensor.

    Given a (BATCH x L X NUM_PREDICTED_ANGLES ) tensor, returns (BATCH X
    L X NUM_PREDICTED_ANGLES) tensor. Performs atan2 transformation from sin
    and cos values.

    Args:
        t (torch.tensor): Tensor of shape (batch, L, NUM_ANGLES, 2).

    Returns:
        torch.tensor: Tensor of angles with the last two dimensions reduced
        via atan2.
    """
    t = t.view(t.shape[0], -1, NUM_ANGLES, 2)
    t_cos = t[:, :, :, 0]
    t_sin = t[:, :, :, 1]
    t = torch.atan2(t_sin, t_cos)
    return t


def trig_transform(t):
    """Expand the last dimension of an angle tensor to have sin/cos values.

    Args:
        t (torch.tensor): Angle tensor with shape (batch x L x num_angle)

    Raises:
        ValueError: if tensor shape is not recognized.

    Returns:
        torch.tensor: Angle tensor with shape (batch x L x num_angle x 2).
    """
    new_t = torch.zeros(*t.shape, 2)
    if len(new_t.shape) == 4:
        new_t[:, :, :, 0] = torch.cos(t)
        new_t[:, :, :, 1] = torch.sin(t)
    else:
        raise ValueError("trig_transform function is only defined for "
                         "(batch x L x num_angle) tensors.")
    return new_t


def compare_pdb_files(file1, file2):
    """Returns the RMSD between two PDB files of the same protein.

    Args:
        file1 (str): Path to first PDB file.
        file2 (str): Path to second PDB file. Must be the same protein as in file1.

    Returns:
        float: Root Mean Squared Deviation (RMSD) between the two structures.
    """
    s1 = pr.parsePDB(file1)
    s2 = pr.parsePDB(file2)
    transformation = pr.calcTransformation(s1, s2)
    s1_aligned = transformation.apply(s1)
    return pr.calcRMSD(s1_aligned, s2)


def debug_example():
    """A simple example of structure building for debugging."""
    d = torch.load("/home/jok120/protein-transformer/data/proteinnet/casp12_200206_30.pt")
    seq = d["train"]["seq"][70]
    ang = d["train"]["ang"][70]
    sb = StructureBuilder.StructureBuilder(seq, ang)
    sb.build()


def coord_generator(coords, atoms_per_res=14, remove_padding=False):
    """Return a generator to iteratively yield self.atoms_per_res atoms at a time."""
    coord_idx = 0
    while coord_idx < coords.shape[0]:
        _slice = coords[coord_idx:coord_idx + atoms_per_res]
        if remove_padding:
            non_pad_locs = (_slice != GLOBAL_PAD_CHAR).any(axis=1)
            _slice = _slice[non_pad_locs]
        yield _slice
        coord_idx += atoms_per_res


if __name__ == '__main__':
    debug_example()
