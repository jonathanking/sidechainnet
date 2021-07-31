"""Methods for adding hydrogen atoms to amino acid residues."""
import math

import numpy as np
import torch
from sidechainnet.structure.build_info import BB_BUILD_INFO, NUM_COORDS_PER_RES, SC_BUILD_INFO
from sidechainnet.structure.structure import coord_generator
from sidechainnet.utils.measure import GLOBAL_PAD_CHAR
from sidechainnet.utils.sequence import ONE_TO_THREE_LETTER_MAP

NUM_COORDS_PER_RES_W_HYDROGENS = 24

METHYL_ANGLE = 109.5
METHYL_LEN = 1.09

METHYLENE_ANGLE = np.rad2deg(0.61656)  # Check - works
METHYLENE_LEN = 1.09

SP3_LEN = 1.09

THIOL_ANGLE = 109.5
THIOL_LEN = 1.336
HYDROXYL_LEN = 0.96

AMIDE_LEN = 1.01  # Backbone
METHINE_LEN = 1.08

AMINE_ANGLE = np.deg2rad(120)
AMINE_LEN = 1.01

OXT_LEN = BB_BUILD_INFO['BONDLENS']['c-oh']

# yapf: disable
HYDROGEN_NAMES = {
    'ALA': ['H', 'HA', 'HB1', 'HB2', 'HB3'],
    'ARG': ['H', 'HA', 'HB2', 'HB3', 'HD2', 'HD3', 'HG2', 'HG3', 'HE', 'HH11', 'HH12', 'HH21', 'HH22'],
    'ASN': ['H', 'HA', 'HB2', 'HB3', 'HD21', 'HD22'],
    'ASP': ['H', 'HA', 'HB2', 'HB3'],
    'CYS': ['H', 'HA', 'HB2', 'HB3', 'HG'],
    'GLN': ['H', 'HA', 'HB2', 'HB3', 'HG2', 'HG3', 'HE21', 'HE22'],
    'GLU': ['H', 'HA', 'HB2', 'HB3', 'HG2', 'HG3'],
    'GLY': ['H', 'HA2', 'HA3'],
    'HIS': ['H', 'HA', 'HB2', 'HB3', 'HE1', 'HD2', 'HD1'],  # CB, CB, CE1, CD2, ND1/HE1
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


class HydrogenBuilder(object):
    """A class for adding Hydrogen positions to set of coordinates."""

    def __init__(self, seq, coords):
        """Create a Hydrogen builder for a protein.

        Args:
            seq (str): Protein sequence.
            coords (numpy.ndarray, torch.tensor): Coordinate set for a protein that does
                not yet contain Hydrogens.
        """
        from sidechainnet.structure.PdbBuilder import ATOM_MAP_14

        self.seq = seq
        self.coords = coords
        self.mode = "torch" if isinstance(coords, torch.Tensor) else "numpy"
        self.is_numpy = self.mode == "numpy"
        self.atom_map = ATOM_MAP_14
        self.terminal_atoms = {"H2": None, "H3": None, "OXT": None}
        self.empty_coord = np.zeros((3)) if self.is_numpy else torch.zeros(3)

        self.norm = np.linalg.norm if self.is_numpy else torch.norm
        self.cross = np.cross if self.is_numpy else torch.cross
        self.dot = np.dot if self.is_numpy else torch.matmul
        self.eye = np.eye if self.is_numpy else torch.eye
        self.ones = np.ones if self.is_numpy else torch.ones
        self.sqrt = math.sqrt if self.is_numpy else torch.sqrt
        self.stack = np.stack if self.is_numpy else torch.stack
        self.array = np.asarray if self.is_numpy else torch.tensor
        self.concatenate = np.concatenate if self.is_numpy else torch.cat

    def build_hydrogens(self):
        """Add hydrogens to internal protein structure."""
        # TODO assumes only one continuous chain (and 1 set of N & C terminals)
        coords = coord_generator(self.coords, NUM_COORDS_PER_RES, remove_padding=True)
        new_coords = []
        prev_res_atoms = None
        for i, (aa, crd) in enumerate(zip(self.seq, coords)):
            # Add empty hydrogen coordinates for missing residues
            if crd.sum() == 0:
                new_coords.append(self.ones((NUM_COORDS_PER_RES_W_HYDROGENS, 3))*0)
                prev_res_atoms = None
                continue
            # Create an organized mapping from atom name to Catesian coordinates
            d = {name: xyz for (name, xyz) in zip(self.atom_map[aa], crd)}
            atoms = AtomHolder(d, default=None)  # default=self.empty_coord would allow building hydrogens from missing coords
            # Generate hydrogen positions as array/tensor
            hydrogen_positions = self.get_hydrogens_for_res(
                aa,
                atoms,
                prev_res_atoms,
                n_terminal=i == 0,
                c_terminal=i == len(self.seq) - 1)
            # Append Hydrogens immediately after heavy atoms, followed by PADs to L=24
            new_coords.append(self.concatenate((crd, hydrogen_positions)))
            prev_res_atoms = atoms
        self.reduced_coords = self.concatenate(new_coords)
        return self.reduced_coords

    # Utility functions
    def M(self, axis, theta):
        """Create rotation matrix with angle theta around a given axis.

        From https://stackoverflow.com/questions/6802577/rotation-of-3d-vector.
        """
        axis = axis / self.sqrt(self.dot(axis, axis))
        a = np.cos(theta / 2.0)
        b, c, d = -axis * np.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        return self.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                           [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                           [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

    def scale(self, vector, target_len, v_len=None):
        """Scale a vector to match a given target length."""
        if v_len is None:
            v_len = self.norm(vector)
        return vector / v_len * target_len

    # The following base functions are utilized for residues consisting of >=1 H geoms.
    # There are 6 base hydrogen geometries:
    #    1. Methyl         (CH3)
    #    2. Methylene      (R-CH2-R)
    #    3. Single-sp3     (N-CAH-C)
    #    4. Hydroxyl/thiol (O-H1, S-H1)
    #    5. Methine/Amide  (C=CH1-C, C-NH1-C)
    #    6. Amine          (H2)

    def get_methyl_hydrogens(self, carbon, prev1, prev2, use_amine_length=False):
        """Place methyl (H3) hydrogens on a Carbon atom. Also supports N-terminal amines.

        Ex: Alanine: carbon, prev1, prev2 are CB, CA, N.
        """
        length = METHYL_LEN if not use_amine_length else AMINE_LEN

        # Define local vectors extending from CA
        N = prev2 - prev1
        CB = carbon - prev1

        # Define perpendicular vector
        PV = self.cross(CB, N)
        R109 = self.M(PV, np.deg2rad(METHYL_ANGLE))  # Rotate abt PV by 109.5 (tetrahed.)

        # Define Hydrogen extending from carbon
        H1 = self.dot(R109, -CB)  # Place Hydrogen by rotating C along perpendicular axis
        H1 = self.scale(H1, length)

        R120 = self.M(CB, 2 * np.pi / 3)
        H2 = self.dot(R120, H1)  # Place 2nd Hydrogen by rotating previous H 120 deg
        H3 = self.dot(R120, H2)  # Place 3rd Hydrogen by rotating previous H 120 deg

        H1 += prev1 + CB
        H2 += prev1 + CB
        H3 += prev1 + CB

        return [H1, H2, H3]

    def get_methylene_hydrogens(self, r1, carbon, r2):
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
        PV = self.cross(R1, R2)
        axis = R2 - R1

        # Place first hydrogen
        R = self.M(axis, METHYLENE_ANGLE)
        H1 = self.dot(R, PV)
        vector_len = self.norm(H1)
        H1 = self.scale(vector=H1, target_len=METHYLENE_LEN, v_len=vector_len)

        # Place second hydrogen
        R = self.M(axis, -METHYLENE_ANGLE)
        H2 = self.dot(R, -PV)
        H2 = self.scale(vector=H2, target_len=METHYLENE_LEN, v_len=vector_len)

        # Return to original position
        H1 += carbon
        H2 += carbon

        return [H1, H2]

    def get_single_sp3_hydrogen(self, center, R1, R2, R3):
        H1 = self.scale(-(R1 + R2 + R3 - (3 * center)), target_len=SP3_LEN)
        return H1 + center

    def get_thiol_hydrogen(self, oxy_sulfur, prev1, prev2, thiol=True):
        if thiol:
            length = THIOL_LEN
        else:
            length = HYDROXYL_LEN

        # Define local vectors
        OS = oxy_sulfur - prev1
        P2 = prev2 - prev1

        # Define perpendicular vector  to other
        PV = self.cross(OS, P2)

        # Define rotation matrices
        # Rotate around PV by 109.5 (tetrahedral)
        RP = self.M(PV, np.pi - np.deg2rad(THIOL_ANGLE))

        # Define Hydrogens
        H1 = self.dot(RP, OS)  # Rotate by tetrahedral angle only
        H1 = self.scale(H1, length)
        H1 += OS

        return H1 + prev1

    def get_amide_methine_hydrogen(self, R1, center, R2, amide=True, oxt=False):
        """Create a planar methine/amide hydrogen (or OXT) given two adjacent atoms."""
        length = AMIDE_LEN if amide else METHINE_LEN
        length = OXT_LEN if oxt else length

        # Define local vectors
        A, B, C = (v - center for v in (R1, center, R2))

        H1 = self.scale(-(A + B + C), target_len=length)
        return H1 + center

    def get_amine_hydrogens(self, nitrogen, prev1, prev2):
        # Define local vectors
        N = nitrogen - prev1
        P2 = prev2 - prev1

        PV = self.cross(N, P2)

        # Place first hydrogen
        R = self.M(PV, -AMINE_ANGLE)  # Rotate around perpendicular axis
        H1 = self.dot(R, -N)
        vector_len = self.norm(H1)
        H1 = self.scale(vector=H1, target_len=AMINE_LEN, v_len=vector_len)

        # Rotate the previous vector around the same axis by another 120 degrees
        H2 = self.dot(R, H1)
        H2 = self.scale(vector=H2, target_len=AMINE_LEN, v_len=vector_len)

        H1 += prev1 + N
        H2 += prev1 + N

        return [H1, H2]

    def pad_hydrogens(self, resname, hydrogens):
        """Pad hydrogen list with empty vectors to the correct length for a given res."""

        pad_vec = [GLOBAL_PAD_CHAR * self.ones(3)]
        n_heavy_atoms = sum(
            [True if an != "PAD" else False for an in self.atom_map[resname]])
        n_pad = NUM_COORDS_PER_RES_W_HYDROGENS - n_heavy_atoms - len(hydrogens)
        hydrogens.extend(pad_vec * (n_pad))
        return hydrogens

    def ala(self, c):
        """Return list of hydrogen positions for ala.

        Positions correspond to the following hydrogens:
            [HB1, HB2, HB3]
        """
        # HB1, HB2, HB3
        hs = self.get_methyl_hydrogens(carbon=c.CB, prev1=c.CA, prev2=c.N)
        return hs

    def arg(self, c):
        """Return list of hydrogen positions for arg.

        Positions correspond to the following hydrogens:
            [HB2, HB3, HD2, HD3, HG2, HG3, HE, HH11, HH12, HH21, HH22]
        """
        hs = []
        # Methylene hydrogens: [HB2, HB3, HD2, HD3, HG2, HG3]
        for R1, carbon, R2 in [
            (c.CA, c.CB, c.CG),
            (c.CG, c.CD, c.NE),
            (c.CB, c.CG, c.CD),
        ]:
            hs.extend(self.get_methylene_hydrogens(R1, carbon, R2))
        # Amide hydrogen: HE
        hs.append(self.get_amide_methine_hydrogen(c.CD, c.NE, c.CZ, amide=True))
        # Amine hydrogens: [HH11, HH12, HH21, HH22]
        hs.extend(self.get_amine_hydrogens(c.NH1, c.CZ, c.NE))
        hs.extend(self.get_amine_hydrogens(c.NH2, c.CZ, c.NE))

        return hs

    def asn(self, c):
        """Return list of hydrogen positions for asn.

        Positions correspond to the following hydrogens:
            [HB2, HB3, HD21, HD22]
        """
        hs = []
        # Methylene: [HB2, HB3]
        hs.extend(self.get_methylene_hydrogens(c.CA, c.CB, c.CG))
        # Amine: [HD21, HD22]
        hs.extend(self.get_amine_hydrogens(c.ND2, c.CG, c.CB))
        return hs

    def asp(self, c):
        """Return list of hydrogen positions for asp.

        Positions correspond to the following hydrogens:
            [HB2, HB3]
        """
        hs = []
        # Methylene: [HB2, HB3]
        hs.extend(self.get_methylene_hydrogens(c.CA, c.CB, c.CG))
        return hs

    def cys(self, c):
        """Return list of hydrogen positions for cys.

        Positions correspond to the following hydrogens:
            [HB2, HB3, HG]
        """
        hs = []
        # Methylene: [HB2, HB3]
        hs.extend(self.get_methylene_hydrogens(c.CA, c.CB, c.SG))
        # Thiol: HG
        hs.append(self.get_thiol_hydrogen(c.SG, c.CB, c.CA))
        return hs

    def gln(self, c):
        """Return list of hydrogen positions for gln.

        Positions correspond to the following hydrogens:
            ['HB2', 'HB3', 'HG2', 'HG3', 'HE21', 'HE22']
        """
        hs = []
        # Methylene: [HB2, HB3]
        hs.extend(self.get_methylene_hydrogens(c.CA, c.CB, c.CG))
        # Methylene: [HG2, HG3]
        hs.extend(self.get_methylene_hydrogens(c.CB, c.CG, c.CD))
        # Amine: [HE21, HE22]
        hs.extend(self.get_amine_hydrogens(c.NE2, c.CD, c.CG))
        return hs

    def glu(self, c):
        """Return list of hydrogen positions for glu.

        Positions correspond to the following hydrogens:
            [HB2, HB3, HG2, HG3]
        """
        hs = []
        # Methylene: [HB2, HB3]
        hs.extend(self.get_methylene_hydrogens(c.CA, c.CB, c.CG))
        # Methylene: [HG2, HG3]
        hs.extend(self.get_methylene_hydrogens(c.CB, c.CG, c.CD))
        return hs

    def gly(self, c):
        """Return list of hydrogen positions for gly.

        Positions correspond to the following hydrogens:
            [HA2, HA3]
        """
        return self.get_methylene_hydrogens(c.N, c.CA, c.C)

    def his(self, c):
        """Return list of hydrogen positions for his.

        Positions correspond to the following hydrogens:
            [HB2, HB3, HE1, HD2, HD1] # CB, CB, CE1, CD2, (ND1/HD1)
        """
        hs = []
        # Methylene: [HB2, HB3]
        hs.extend(self.get_methylene_hydrogens(c.CA, c.CB, c.CG))
        # Methine x2: [HE1, HD2]
        hs.append(self.get_amide_methine_hydrogen(c.ND1, c.CE1, c.NE2, amide=False))
        hs.append(self.get_amide_methine_hydrogen(c.CG, c.CD2, c.NE2, amide=False))
        # Amide: [HD1]
        hs.append(self.get_amide_methine_hydrogen(c.CG, c.ND1, c.CE1, amide=True))
        return hs

    def ile(self, c):
        """Return list of hydrogen positions for ile.

        Positions correspond to the following hydrogens:
            [HB, HD11, HD12, HD13, HG12, HG13, HG21, HG22, HG23]
        """
        hs = []
        # HB
        hs.append(self.get_single_sp3_hydrogen(c.CB, c.CA, c.CG1, c.CG2))
        # HD11, HD12, HD13
        hs.extend(self.get_methyl_hydrogens(c.CD1, c.CG1, c.CB))
        # Methylene: HG12, HG13
        hs.extend(self.get_methylene_hydrogens(c.CB, c.CG1, c.CD1))
        # Methyl: HG21, HG22, HG23
        hs.extend(self.get_methyl_hydrogens(c.CG2, c.CB, c.CA))
        return hs

    def leu(self, c):
        """Return list of hydrogen positions for leu.

        Positions correspond to the following hydrogens:
            [HB2, HB3, HD11, HD12, HD13, HD21, HD22, HD23, HG]
        """
        hs = []
        # Methylene: [HB2, HB3]
        hs.extend(self.get_methylene_hydrogens(c.CA, c.CB, c.CG))
        # Methyl: HD11, HD12, HD13
        hs.extend(self.get_methyl_hydrogens(c.CD1, c.CG, c.CB))
        # Methyl: HD21, HD22, HD23
        hs.extend(self.get_methyl_hydrogens(c.CD2, c.CG, c.CB))
        # SP3: HG
        hs.append(self.get_single_sp3_hydrogen(c.CG, c.CB, c.CD1, c.CD2))
        return hs

    def lys(self, c):
        """Return list of hydrogen positions for lys.

        Positions correspond to the following hydrogens:
            [HB2, HB3, HD2, HD3, HE2, HE3, HG2, HG3, HZ1, HZ2, HZ3]
        """
        hs = []
        # Methylene x4: [HB2, HB3], [HD2, HD3], [HE2, HE3], [HG2, HG3] ie (CB, CD, CE, CG)
        for r1, carbon, r2 in ((c.CA, c.CB, c.CG), (c.CG, c.CD, c.CE), (c.CD, c.CE, c.NZ),
                               (c.CB, c.CG, c.CD)):
            hs.extend(self.get_methylene_hydrogens(r1, carbon, r2))
        # NH3: HZ1, HZ2, HZ3
        hs.extend(self.get_methyl_hydrogens(c.NZ, c.CE, c.CD))
        return hs

    def met(self, c):
        """Return list of hydrogen positions for met.

        Positions correspond to the following hydrogens:
            [HB2, HB3, HE1, HE2, HE3, HG2, HG3]
        """
        hs = []
        # Methylene: [HB2, HB3]
        hs.extend(self.get_methylene_hydrogens(c.CA, c.CB, c.CG))
        # Methyl: HE1, HE2, HE3
        hs.extend(self.get_methyl_hydrogens(c.CE, c.SD, c.CG))
        # Methylene: HG2, HG3
        hs.extend(self.get_methylene_hydrogens(c.CB, c.CG, c.SD))
        return hs

    def phe(self, c):
        """Return list of hydrogen positions for phe.

        Positions correspond to the following hydrogens:
            [HB2, HB3, HD1, HD2, HE1, HE2, HZ]
        """
        hs = []
        # Methylene: HB2, HB3
        hs.extend(self.get_methylene_hydrogens(c.CA, c.CB, c.CG))
        # Methine x 5: HD1, HD2, HE1, HE2, HZ
        for r1, center, r2 in ((c.CG, c.CD1, c.CE1), (c.CG, c.CD2, c.CE2),
                               (c.CD1, c.CE1, c.CZ), (c.CD2, c.CE2, c.CZ), (c.CE1, c.CZ,
                                                                            c.CE2)):
            hs.append(self.get_amide_methine_hydrogen(r1, center, r2, amide=False))
        return hs

    def pro(self, c):
        """Return list of hydrogen positions for pro.

        Positions correspond to the following hydrogens:
            [HB2, HB3, HD2, HD3, HG2, HG3]
        """
        hs = []
        # Methylene x6: [HB2, HB3], [HD2, HD3], [HG2, HG3]
        hs.extend(self.get_methylene_hydrogens(c.CA, c.CB, c.CG))
        hs.extend(self.get_methylene_hydrogens(c.N, c.CD, c.CG))
        hs.extend(self.get_methylene_hydrogens(c.CD, c.CG, c.CB))
        return hs

    def ser(self, c):
        """Return list of hydrogen positions for ser.

        Positions correspond to the following hydrogens:
            [HB2, HB3, HG]
        """
        hs = []
        # Methylene: [HB2, HB3]
        hs.extend(self.get_methylene_hydrogens(c.CA, c.CB, c.OG))
        # Hydroxyl: HG
        hs.append(self.get_thiol_hydrogen(c.OG, c.CB, c.CA, thiol=False))
        return hs

    def thr(self, c):
        """Return list of hydrogen positions for thr.

        Positions correspond to the following hydrogens:
            [HB, HG1, HG21, HG22, HG23]
        """
        hs = []
        # SP3: HB
        hs.append(self.get_single_sp3_hydrogen(c.CB, c.CA, c.OG1, c.CG2))
        # Hydroxyl: HG1
        hs.append(self.get_thiol_hydrogen(c.OG1, c.CB, c.CA, thiol=False))
        # Methyl: HG21, HG22, HG23
        hs.extend(self.get_methyl_hydrogens(c.CG2, c.CB, c.CA))
        return hs

    def trp(self, c):
        """Return list of hydrogen positions for trp.

        Positions correspond to the following hydrogens:
            [HB2, HB3, HD1, HE1, HE3, HH2, HZ2, HZ3]
        """
        hs = []
        # Methylene: [HB2, HB3]
        hs.extend(self.get_methylene_hydrogens(c.CA, c.CB, c.CG))
        # Methine: HD1
        hs.append(self.get_amide_methine_hydrogen(c.CG, c.CD1, c.NE1, amide=False))
        # Amide/methine: HE1
        hs.append(self.get_amide_methine_hydrogen(c.CD1, c.NE1, c.CE2, amide=True))
        # Methine: HE3
        hs.append(self.get_amide_methine_hydrogen(c.CD2, c.CE3, c.CZ3, amide=False))
        # Methine: HH2
        hs.append(self.get_amide_methine_hydrogen(c.CZ3, c.CH2, c.CZ2, amide=False))
        # Methine HZ2
        hs.append(self.get_amide_methine_hydrogen(c.CE2, c.CZ2, c.CH2, amide=False))
        # Methine: HZ3
        hs.append(self.get_amide_methine_hydrogen(c.CE3, c.CZ3, c.CH2, amide=False))
        return hs

    def tyr(self, c):
        """Return list of hydrogen positions for tyr.

        Positions correspond to the following hydrogens:
            [HB2, HB3, HD1, HD2, HE1, HE2, HH]
        """
        hs = []
        # Methylene: [HB2, HB3]
        hs.extend(self.get_methylene_hydrogens(c.CA, c.CB, c.CG))
        # Methine x4: HD1, HD2, HE1, HE2
        hs.append(self.get_amide_methine_hydrogen(c.CG, c.CD1, c.CE1, amide=False))
        hs.append(self.get_amide_methine_hydrogen(c.CG, c.CD2, c.CE2, amide=False))
        hs.append(self.get_amide_methine_hydrogen(c.CD1, c.CE1, c.CZ, amide=False))
        hs.append(self.get_amide_methine_hydrogen(c.CD2, c.CE2, c.CZ, amide=False))
        # Hydroxyl: HH
        hs.append(self.get_thiol_hydrogen(c.OH, c.CZ, c.CE1, thiol=False))
        return hs

    def val(self, c):
        """Return list of hydrogen positions for val.

        Positions correspond to the following hydrogens:
            [HB, HG11, HG12, HG13, HG21, HG22, HG23]
        """
        hs = []
        # SP3: HB
        hs.append(self.get_single_sp3_hydrogen(c.CB, c.CA, c.CG1, c.CG2))
        # Methyl  x2: [HG11, HG12, HG13], [HG21, HG22, HG23]
        hs.extend(self.get_methyl_hydrogens(c.CG1, c.CB, c.CA))
        hs.extend(self.get_methyl_hydrogens(c.CG2, c.CB, c.CA))
        return hs

    def get_hydrogens_for_res(self,
                              resname,
                              c,
                              prevc,
                              n_terminal=False,
                              c_terminal=False):
        """Return a padded array of hydrogens for a given res name & atom coord tuple."""
        hs = []
        # Special Cases
        if n_terminal:
            h, h2, h3 = self.get_methyl_hydrogens(c.N, c.CA, c.C, use_amine_length=True)
            self.terminal_atoms.update({"H2": h2, "H3": h3})
            hs.append(h)  # Used as normal amine hydrogen, H
        if c_terminal:
            oxt = self.get_amide_methine_hydrogen(c.CA, c.C, c.O, oxt=True)
            self.terminal_atoms.update({"OXT": oxt})
        # All amino acids except proline have an amide-hydrogen along the backbone
        if prevc and resname != "P":
            hs.append(self.get_amide_methine_hydrogen(prevc.C, c.N, c.CA, amide=True))
        # If the amino acid is not Glycine, we can add an sp3-hybridized H to CA
        if resname != "G":
            cah = self.get_single_sp3_hydrogen(center=c.CA, R1=c.N, R2=c.C, R3=c.CB)
            hs.append(cah)

        # Now, we can add the remaining unique hydrogens for each amino acid
        if resname == "A":
            hs.extend(self.ala(c))
        elif resname == "R":
            hs.extend(self.arg(c))
        elif resname == "N":
            hs.extend(self.asn(c))
        elif resname == "D":
            hs.extend(self.asp(c))
        elif resname == "C":
            hs.extend(self.cys(c))
        elif resname == "E":
            hs.extend(self.glu(c))
        elif resname == "Q":
            hs.extend(self.gln(c))
        elif resname == "G":
            hs.extend(self.gly(c))
        elif resname == "H":
            hs.extend(self.his(c))
        elif resname == "I":
            hs.extend(self.ile(c))
        elif resname == "L":
            hs.extend(self.leu(c))
        elif resname == "K":
            hs.extend(self.lys(c))
        elif resname == "M":
            hs.extend(self.met(c))
        elif resname == "F":
            hs.extend(self.phe(c))
        elif resname == "P":
            hs.extend(self.pro(c))
        elif resname == "S":
            hs.extend(self.ser(c))
        elif resname == "T":
            hs.extend(self.thr(c))
        elif resname == "W":
            hs.extend(self.trp(c))
        elif resname == "Y":
            hs.extend(self.tyr(c))
        elif resname == "V":
            hs.extend(self.val(c))

        return self.stack(self.pad_hydrogens(resname, hs), 0)


class AtomHolder(dict):
    """A defaultdict that also allows keys to be accessed like attributes.

    Used to map atom names to their coordinates. If an atom doesn't exist, returns a 
    default value. Inspired by:
    https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary.
    """

    def __init__(self, dictionary, default):
        for k, v in dictionary.items():
            self[k] = v
        self.default = default

    def __getattr__(self, name):
        if name in self:
            return self[name]
        elif self.default:
            return self.default.copy()
        else:
            raise AttributeError(f"No attribute named {name}.")

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(AtomHolder, self).__setitem__(key, value)
        self.__dict__.update({key: value})


ATOM_MAP_24 = {}
for one_letter, three_letter in ONE_TO_THREE_LETTER_MAP.items():
    ATOM_MAP_24[one_letter] = ["N", "CA", "C", "O"] + list(
        SC_BUILD_INFO[ONE_TO_THREE_LETTER_MAP[one_letter]]["atom-names"])
    ATOM_MAP_24[one_letter].extend(HYDROGEN_NAMES[three_letter])
    ATOM_MAP_24[one_letter].extend(["PAD"] * (24 - len(ATOM_MAP_24[one_letter])))
