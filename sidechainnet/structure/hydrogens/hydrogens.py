from sidechainnet.structure.build_info import SC_BUILD_INFO
from sidechainnet.utils.sequence import ONE_TO_THREE_LETTER_MAP
import numpy as np
from numpy import cross, eye, dot
from scipy.linalg import expm, norm

NUM_COORDS_PER_RES_W_HYDROGENS = 24

METHYL_ANGLE = 109.5
METHYL_LEN = 1.01

METHYLENE_ANGLE = np.rad2deg(0.61656)
METHYLENE_LEN = 1.09

SP3_LEN = 1.09

THIOL_ANGLE = 109.5
THIOL_LEN = 1.09

AMIDE_LEN = 1.09
METHINE_LEN = 1.09

AMINE_ANGLE = 2 * np.pi / 3
AMINE_LEN = 1.09

# yapf: disable
HYDROGEN_NAMES = {
    'ALA': ['H', 'HA', 'HB1', 'HB2', 'HB3'],
    'ARG': ['H', 'HA', 'HB2', 'HB3', 'HD2', 'HD3', 'HG2', 'HG3', 'HE', 'HH11', 'HH12', 'HH21', 'HH22'],
    'ASN': ['H', 'HA', 'HB2', 'HB3', 'HD21', 'HD22'],
    'ASP': ['H', 'HA', 'HB2', 'HB3'],
    'CYS': ['H', 'HA', 'HB2', 'HB3', 'HG'],
    'GLN': ['H', 'HA', 'HB2', 'HB3', 'HE21', 'HE22', 'HG2', 'HG3'],
    'GLU': ['H', 'HA', 'HB2', 'HB3', 'HG2', 'HG3'],
    'GLY': ['H', 'HA2', 'HA3'],
    'HIS': ['H', 'HA', 'HB2', 'HB3', 'HD2', 'HE1'],
    'ILE': ['H', 'HA', 'HB', 'HD11', 'HD12', 'HD13', 'HG12', 'HG13', 'HG21', 'HG22', 'HG23'],
    'LEU': ['H', 'HA', 'HB2', 'HB3', 'HD11', 'HD12', 'HD13', 'HD21', 'HD22', 'HD23', 'HG'],
    'LYS': ['H', 'HA', 'HB2', 'HB3', 'HD2', 'HD3', 'HE2', 'HE3', 'HG2', 'HG3', 'HZ1', 'HZ2', 'HZ3'],
    'MET': ['H', 'HA', 'HB2', 'HB3', 'HE1', 'HE2', 'HE3', 'HG2', 'HG3'],
    'PHE': ['H', 'HA', 'HB2', 'HB3', 'HD1', 'HD2', 'HE1', 'HE2', 'HZ'],
    'PRO': ['HA', 'HB2', 'HB3', 'HD2', 'HD3', 'HG2', 'HG3'],
    'SER': ['H', 'HA', 'HB2', 'HB3', 'HG'],
    'THR': ['H', 'HA', 'HB', 'HG1', 'HG21', 'HG22', 'HG23'],
    'TRP': ['H', 'HA', 'HB2', 'HB3', 'HD1', 'HE1', 'HE3', 'HH2', 'HZ2', 'HZ3'],
    'TYR': ['H', 'HA', 'HB2', 'HB3', 'HD1', 'HD2', 'HE1', 'HE2', 'HH'],
    'VAL': ['H', 'HA', 'HB', 'HG11', 'HG12', 'HG13', 'HG21', 'HG22', 'HG23']
}
# yapf: enable


# Utility functions
def M(axis, theta):
    """Create rotation matrix with angle theta around a given axis.

    From https://stackoverflow.com/questions/6802577/rotation-of-3d-vector.
    """
    return expm(cross(eye(3), axis / norm(axis) * theta))


def scale(vector, target_len, v_len=None):
    """Scale a vector to match a given target length."""
    if v_len is None:
        v_len = np.linalg.norm(vector)
    return vector / v_len * target_len


# The following base functions are utilized for residues consisting of >=1 H geometries.
# There are 6 base hydrogen geometries:
#    1. Methyl         (CH3)
#    2. Methylene      (R-CH2-R)
#    3. Single-sp3     (N-CAH-C)
#    4. Hydroxyl/thiol (O-H1, S-H1)
#    5. Methine/Amide  (C=CH1-C, C-NH1-C)
#    6. Amine          (H2)


def get_methyl_hydrogens(carbon, prev1, prev2):
    """Place methyl (H3) hydrogens on a Carbon atom.

    Ex: Alanine: carbon, prev1, prev2 are CB, CA, N.
    """
    # Define local vectors extending from CA
    N = prev2 - prev1
    CB = carbon - prev1

    # Define perpendicular vector
    PV = np.cross(CB, N)
    R109 = M(PV, np.deg2rad(METHYL_ANGLE))  # Rotate around PV by 109.5 (tetrahedral)

    # Define Hydrogen extending from carbon
    H1 = dot(R109, -CB)  # Place Hydrogen by rotating C along perpendicular axis
    H1 = scale(H1, METHYL_LEN)

    R120 = M(CB, 2 * np.pi / 3)
    H2 = dot(R120, H1)  # Place 2nd Hydrogen by rotating previous H 120 deg
    H3 = dot(R120, H2)  # Place 3rd Hydrogen by rotating previous H 120 deg

    H1 += prev1 + CB
    H2 += prev1 + CB
    H3 += prev1 + CB

    return [H1, H2, H3]


def get_methylene_hydrogens(r1, carbon, r2):
    """Place methylene hydrogens (R1-CH2-R2) on central Carbon.

    Args:
        r1: First atom vector.
        carbon: Second atom vector (Carbon needing hydrogens).
        r2: Third atom vector.

    Returns:
        Tuple: Hydrogens extending from central Carbon.
    """
    # Define local vectors
    R1 = r1 - carbon
    R2 = r2 - carbon

    # Create perpendicular vector
    PV = np.cross(R1, R2)
    axis = R2 - R1

    # Place first hydrogen
    R = M(axis, METHYLENE_ANGLE)
    H1 = dot(R, PV)
    vector_len = np.linalg.norm(H1)
    H1 = scale(vector=H1, target_len=METHYLENE_LEN, v_len=vector_len)

    # Place second hydrogen
    R = M(axis, -METHYLENE_ANGLE)
    H2 = dot(R, -PV)
    H2 = scale(vector=H2, target_len=METHYLENE_LEN, v_len=vector_len)

    # Return to original position
    H1 += carbon
    H2 += carbon

    return [H1, H2]


def get_single_sp3_hydrogen(center, R1, R2, R3):
    H1 = scale(-(R1 + R2 + R3 - (3 * center)), target_len=SP3_LEN)
    return H1 + center


def get_thiol_hydrogen(oxy_sulfur, prev1, prev2):
    # Define local vectors
    OS = oxy_sulfur - prev1
    P2 = prev2 - prev1

    # Define perpendicular vector  to other
    PV = np.cross(OS, P2)

    # Define rotation matrices
    RP = M(PV, np.pi - np.deg2rad(THIOL_ANGLE))  # Rotate around PV by 109.5 (tetrahedral)
    RQ = M(OS, np.pi / 2)  # Rotate around thiol axis by 1/4 turn

    # Define Hydrogens
    H1 = dot(RQ, dot(RP, OS))  # Place Hydrogen by rotating OS vec twice
    H1 = scale(H1, THIOL_LEN)
    H1 += OS

    return H1 + prev1


def get_amide_methine_hydrogen(R1, center, R2, amide=True):
    length = AMIDE_LEN if amide else METHINE_LEN

    # Define local vectors
    A, B, C = (v - center for v in (R1, center, R2))

    H1 = scale(-(A + B + C), target_len=length)
    return H1 + center


def get_amine_hydrogens(nitrogen, prev1, prev2):
    # Define local vectors
    N = nitrogen - prev1
    P2 = prev2 - prev1

    PV = np.cross(N, P2)

    # Place first hydrogen
    R = M(PV, -AMINE_ANGLE)  # Rotate around perpendicular axis
    H1 = dot(R, -N)
    vector_len = np.linalg.norm(H1)
    H1 = scale(vector=H1, target_len=AMINE_LEN, v_len=vector_len)

    # Rotate the previous vector around the same axis by another 120 degrees
    H2 = dot(R, H1)
    H2 = scale(vector=H2, target_len=AMINE_LEN, v_len=vector_len)

    H1 += prev1 + N
    H2 += prev1 + N

    return [H1, H2]


ATOM_MAP_24 = {}
for one_letter, three_letter in ONE_TO_THREE_LETTER_MAP.items():
    ATOM_MAP_24[one_letter] = ["N", "CA", "C", "O"] + list(
        SC_BUILD_INFO[ONE_TO_THREE_LETTER_MAP[one_letter]]["atom-names"])
    ATOM_MAP_24[one_letter].extend(HYDROGEN_NAMES[three_letter])
    ATOM_MAP_24[one_letter].extend(["PAD"] * (24 - len(ATOM_MAP_24[one_letter])))
