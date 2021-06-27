import numpy as np
from numpy import cross, eye, dot
from scipy.linalg import expm, norm

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

    return H1, H2, H3


def get_methylene_hydrogens(R1, carbon, R2):
    """Place methylene hydrogens (R1-CH2-R2) on central Carbon.

    Args:
        R1: First atom vector.
        carbon: Second atom vector (Carbon needing hydrogens).
        R2: Third atom vector.

    Returns:
        Tuple: Hydrogens extending from central Carbon.
    """
    # Define local vectors
    R1 = R1 - carbon
    R2 = R2 - carbon

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

    return H1, H2


def get_single_sp3_hydrogen(center, R1, R2, R3):
    R1 -= center
    R2 -= center
    R3 -= center
    H1 = scale(-(R1 + R2 + R3), target_len=SP3_LEN)
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

    return H1, H2

