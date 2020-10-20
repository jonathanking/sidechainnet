import numpy as np
import torch
from sidechainnet.structure import StructureBuilder
from sidechainnet.utils.sequence import VOCAB


def angles_to_coords(angles, seq, remove_batch_padding=False):
    """
    Convert torsional angles to coordinates.
    """
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
    """ Returns a protein's coordinates generated from its angles and sequence.

    Given a tensor of angles (L x NUM_PREDICTED_ANGLES), produces the entire
    set of cartesian coordinates using the NeRF method, (L x A` x 3),
    where A` is the number of atoms generated (depends on amino acid sequence).
    """
    sb = StructureBuilder.StructureBuilder(input_seq, angles, device)
    return sb.build()


def nerf(a, b, c, l, theta, chi):
    """
    Natural extension reference frame method for placing the 4th atom given
    atoms 1-3 and the relevant angle inforamation. This code was originally
    written by Rohit Bhattacharya (rohit.bhattachar@gmail.com,
    https://github.com/rbhatta8/protein-design/blob/master/nerf.py) and I
    have extended it to work with PyTorch. His original documentation is
    below:

    Nerf method of finding 4th coord (d) in cartesian space
        Params:
            a, b, c : coords of 3 points
            l : bond length between c and d
            theta : bond angle between b, c, d (in degrees)
            chi : dihedral using a, b, c, d (in degrees)
        Returns:
            d: tuple of (x, y, z) in cartesian space
    """
    # calculate unit vectors AB and BC
    if not (-np.pi <= theta <= np.pi):
        raise ValueError(f"theta must be in radians and in [-pi, pi]. theta = {theta}")

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
    # TODO: is the squeezing necessary?
    d = d.unsqueeze(1).to(torch.float32)
    res = c + torch.mm(M, d).squeeze()
    return res.squeeze()


def determine_missing_positions(ang_or_coord_matrix):
    """Uses GLOBAL_PAD_CHAR to determine location of missing atoms or residues."""
    raise NotImplementedError


def deg2rad(angle):
    """
    Converts an angle in degrees to radians.
    """
    return angle * np.pi / 180.


# import sidechainnet.protein.StructureBuilder as StructureBuilder

if __name__ == '__main__':
    d = torch.load("/home/jok120/protein-transformer/data/proteinnet/casp12_200206_30.pt")
    seq = d["train"]["seq"][70]
    ang = d["train"]["ang"][70]
    sb = StructureBuilder.StructureBuilder(seq, ang)
    print(sb.seq_as_str())
    sb.build()
    print("Hi")
    pass
