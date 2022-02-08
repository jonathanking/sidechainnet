"""Methods for adding hydrogen atoms to amino acid residues."""
import math

import numpy as np
from numba import njit
import torch
from sidechainnet.structure.build_info import BB_BUILD_INFO, NUM_COORDS_PER_RES, SC_BUILD_INFO
from sidechainnet.structure.structure import coord_generator
from sidechainnet.utils.measure import GLOBAL_PAD_CHAR
from sidechainnet.utils.sequence import ONE_TO_THREE_LETTER_MAP

NUM_COORDS_PER_RES_W_HYDROGENS = 26

METHYL_ANGLE = torch.tensor(np.deg2rad(109.5))
METHYL_LEN = torch.tensor(1.09)

METHYLENE_ANGLE = torch.tensor(np.rad2deg(0.61656))  # Check - works
METHYLENE_LEN = torch.tensor(1.09)

SP3_LEN = torch.tensor(1.09)

THIOL_ANGLE = torch.tensor(109.5)
THIOL_LEN = torch.tensor(1.336)
HYDROXYL_LEN = torch.tensor(0.96)

AMIDE_LEN = torch.tensor(1.01)  # Backbone
METHINE_LEN = torch.tensor(1.08)

AMINE_ANGLE = torch.tensor(np.deg2rad(120))
AMINE_LEN = torch.tensor(1.01)

OXT_LEN = torch.tensor(BB_BUILD_INFO['BONDLENS']['c-oh'])

RAD120TORCH = torch.tensor(2 * np.pi / 3)
THIOL_ROT = torch.tensor(1.23045715359)

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

HYDROGEN_PARTNERS = {  # Lists the atom each hydrogen (from list above) is connected to
    'ALA': ['N', 'CA', 'CB', 'CB', 'CB'],
    'ARG': ['N', 'CA', 'CB', 'CB', 'CD', 'CD', 'CG', 'CG', 'NE', 'NH1', 'NH1', 'NH2', 'NH2'],
    'ASN': ['N', 'CA', 'CB', 'CB', 'ND2', 'ND2'],
    'ASP': ['N', 'CA', 'CB', 'CB'],
    'CYS': ['N', 'CA', 'CB', 'CB', 'SG'],
    'GLN': ['N', 'CA', 'CB', 'CB', 'CG', 'CG', 'NE2', 'NE2'],
    'GLU': ['N', 'CA', 'CB', 'CB', 'CG', 'CG'],
    'GLY': ['N', 'CA', 'CA'],
    'HIS': ['N', 'CA', 'CB', 'CB', 'CE1', 'CD2', 'ND1'],  # CB, CB, CE1, CD2, ND1/HE1
    'ILE': ['N', 'CA', 'CB', 'CD1', 'CD1', 'CD1', 'CG1', 'CG1', 'CG2', 'CG2', 'CG2'],
    'LEU': ['N', 'CA', 'CB', 'CB', 'CD1', 'CD1', 'CD1', 'CD2', 'CD2', 'CD2', 'CG'],
    'LYS': ['N', 'CA', 'CB', 'CB', 'CD', 'CD', 'CE', 'CE', 'CG', 'CG', 'NZ', 'NZ', 'NZ'],
    'MET': ['N', 'CA', 'CB', 'CB', 'CE', 'CE', 'CE', 'CG', 'CG'],
    'PHE': ['N', 'CA', 'CB', 'CB', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
    'PRO': ['CA', 'CB', 'CB', 'CD', 'CD', 'CG', 'CG'],
    'SER': ['N', 'CA', 'CB', 'CB', 'OG'],
    'THR': ['N', 'CA', 'CB', 'OG1', 'CG2', 'CG2', 'CG2'],
    'TRP': ['N', 'CA', 'CB', 'CB', 'CD', 'NE1', 'CE3', 'CH2', 'CZ2', 'CZ3'],
    'TYR': ['N', 'CA', 'CB', 'CB', 'CD1', 'CD2', 'CE1', 'CE2', 'OH'],
    'VAL': ['N', 'CA', 'CB', 'CG1', 'CG1', 'CG1', 'CG2', 'CG2', 'CG2']
}

HYDROGEN_NAMES_TO_PARTNERS = {}
for resname in HYDROGEN_NAMES.keys():
    if resname not in HYDROGEN_NAMES_TO_PARTNERS:
        HYDROGEN_NAMES_TO_PARTNERS[resname] = {}
    for hname, pname in zip(HYDROGEN_NAMES[resname], HYDROGEN_PARTNERS[resname]):
        HYDROGEN_NAMES_TO_PARTNERS[resname][hname] = pname
    HYDROGEN_NAMES_TO_PARTNERS[resname]["H2"] = HYDROGEN_NAMES_TO_PARTNERS[resname]["H3"] = "N"
# yapf: enable


class HydrogenBuilder(object):
    """A class for adding Hydrogen positions to set of coordinates."""

    def __init__(self, seq, coords, device='cpu'):
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
        self.empty_coord = np.zeros(
            (3)) if self.is_numpy else torch.zeros(3, device=device)
        self.device = device

        self.norm = np.linalg.norm if self.is_numpy else torch.norm
        self.cross = np.cross if self.is_numpy else torch.cross
        self.dot = np.dot if self.is_numpy else torch.matmul
        self.eye = np.eye if self.is_numpy else torch.eye
        self.ones = np.ones if self.is_numpy else torch.ones
        self.zeros = np.zeros if self.is_numpy else torch.zeros
        self.sqrt = math.sqrt if self.is_numpy else torch.sqrt
        self.stack = np.stack if self.is_numpy else torch.stack
        self.array = np.asarray if self.is_numpy else torch.tensor
        self.concatenate = np.concatenate if self.is_numpy else torch.cat

    def build_hydrogens(self):
        """Add hydrogens to internal protein structure."""
        # TODO assumes only one continuous chain (and 1 set of N & C terminals)
        coords = coord_generator(self.coords,
                                 NUM_COORDS_PER_RES,
                                 remove_padding=True,
                                 seq=self.seq)
        new_coords = []
        prev_res_atoms = None
        for i, (aa, crd) in enumerate(zip(self.seq, coords)):
            # Add empty hydrogen coordinates for missing residues
            if crd.sum() == 0:
                new_coords.append(self.ones((NUM_COORDS_PER_RES_W_HYDROGENS, 3)) * 0)
                prev_res_atoms = None
                continue
            # Create an organized mapping from atom name to Catesian coordinates
            d = {name: xyz for (name, xyz) in zip(self.atom_map[aa], crd)}
            atoms = AtomHolder(
                d, default=None
            )  # default=self.empty_coord would allow building hydrogens from missing coords
            # Generate hydrogen positions as array/tensor
            hydrogen_positions = self.get_hydrogens_for_res(
                aa,
                atoms,
                prev_res_atoms,
                n_terminal=i == 0,
                c_terminal=i == len(self.seq) - 1)
            # Append Hydrogens immediately after heavy atoms, followed by PADs to L=26
            new_coords.append(self.concatenate((crd, hydrogen_positions)))
            prev_res_atoms = atoms
        self.reduced_coords = self.concatenate(new_coords)
        return self.reduced_coords

    # Utility functions
    def M(self, axis, theta, posneg=False):
        """Create rotation matrix with angle theta around a given axis.

        From https://stackoverflow.com/questions/6802577/rotation-of-3d-vector.
        """
        if not posneg:
            rot_matrix = _M(axis.detach().cpu().numpy(), theta.detach().cpu().numpy())
            return torch.tensor(rot_matrix, device=self.device)
        else:
            r1, r2 = _M_posneg(axis.detach().cpu().numpy(),
                               theta.detach().cpu().numpy())
            return (torch.tensor(r1, device=self.device),
                    torch.tensor(r2, device=self.device))

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

    def get_methyl_hydrogens(self, carbon, prev1, prev2, length):
        """Place methyl (H3) hydrogens on a Carbon atom. Also supports N-terminal amines.

        Ex: Alanine: carbon, prev1, prev2 are CB, CA, N.
        """
        # Define local vectors extending from CA
        N = prev2 - prev1
        CB = carbon - prev1

        # Define perpendicular vector
        PV = self.cross(CB, N)
        R109 = self.M(PV, METHYL_ANGLE)  # Rotate abt PV by 109.5 (tetrahed.)

        # Define Hydrogen extending from carbon
        H1 = self.dot(R109, -CB)  # Place Hydrogen by rotating C along perpendicular axis
        H1 = self.scale(H1, length)

        R120 = self.M(CB, RAD120TORCH)
        # Place 2nd Hydrogen by rotating prev H 120 deg
        H2 = self.dot(R120, self.clone(H1))
        # Place 3rd Hydrogen by rotating prev H 120 deg
        H3 = self.dot(R120, self.clone(H2))

        # Return to original position
        H1 += carbon
        H2 += carbon
        H3 += carbon

        return [H1, H2, H3]

    def clone(self, x):
        """Clone input iff input is a tensor."""
        if self.is_numpy:
            return x
        else:
            return x.clone()

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
        ROT1, ROT2 = self.M(axis, METHYLENE_ANGLE, posneg=True)
        H1 = self.dot(ROT1, PV)
        vector_len = self.norm(H1)
        H1 = _scale_l(vector=H1, target_len=METHYLENE_LEN, v_len=vector_len)

        # Place second hydrogen
        H2 = self.dot(ROT2, -PV)
        H2 = _scale_l(vector=H2, target_len=METHYLENE_LEN, v_len=vector_len)

        # Return to original position
        H1 += carbon
        H2 += carbon

        return [H1, H2]

    def get_single_sp3_hydrogen(self, center, R1, R2, R3):
        h = _get_single_sp3_hydrogen_help(center, R1, R2, R3)
        H1 = _scale(h, SP3_LEN)
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
        RP = self.M(PV, THIOL_ROT)

        # Define Hydrogens
        H1 = self.dot(RP, OS)  # Rotate by tetrahedral angle only
        H1 = _scale(H1, length)
        H1 += OS

        return H1 + prev1

    def get_amide_methine_hydrogen(self, R1, center, R2, amide=True, oxt=False):
        """Create a planar methine/amide hydrogen (or OXT) given two adjacent atoms."""
        length = AMIDE_LEN if amide else METHINE_LEN
        length = OXT_LEN if oxt else length

        H1 = _get_amide_methine_hydrogen_help(R1, center, R2, length)

        return H1

    def get_amine_hydrogens(self, nitrogen, prev1, prev2):
        # Define local vectors
        N = nitrogen - prev1
        P2 = prev2 - prev1

        PV = self.cross(N, P2)

        # Place first hydrogen
        R = self.M(PV, -AMINE_ANGLE)  # Rotate around perpendicular axis
        H1 = self.dot(R, -N)
        vector_len = self.norm(H1)
        H1 = _scale_l(vector=H1, target_len=AMINE_LEN, v_len=vector_len)

        # Rotate the previous vector around the same axis by another 120 degrees
        H2 = self.dot(R, self.clone(H1))
        H2 = _scale_l(vector=H2, target_len=AMINE_LEN, v_len=vector_len)

        H1 += nitrogen
        H2 += nitrogen

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
        hs = self.get_methyl_hydrogens(carbon=c.CB,
                                       prev1=c.CA,
                                       prev2=c.N,
                                       length=METHYL_LEN)
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
        hs.extend(self.get_methyl_hydrogens(c.CD1, c.CG1, c.CB, length=METHYL_LEN))
        # Methylene: HG12, HG13
        hs.extend(self.get_methylene_hydrogens(c.CB, c.CG1, c.CD1))
        # Methyl: HG21, HG22, HG23
        hs.extend(self.get_methyl_hydrogens(c.CG2, c.CB, c.CA, length=METHYL_LEN))
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
        hs.extend(self.get_methyl_hydrogens(c.CD1, c.CG, c.CB, length=METHYL_LEN))
        # Methyl: HD21, HD22, HD23
        hs.extend(self.get_methyl_hydrogens(c.CD2, c.CG, c.CB, length=METHYL_LEN))
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
        hs.extend(self.get_methyl_hydrogens(c.NZ, c.CE, c.CD, length=METHYL_LEN))
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
        hs.extend(self.get_methyl_hydrogens(c.CE, c.SD, c.CG, length=METHYL_LEN))
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
        hs.extend(self.get_methyl_hydrogens(c.CG2, c.CB, c.CA, length=METHYL_LEN))
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
        hs.extend(self.get_methyl_hydrogens(c.CG1, c.CB, c.CA, length=METHYL_LEN))
        hs.extend(self.get_methyl_hydrogens(c.CG2, c.CB, c.CA, length=METHYL_LEN))
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
            h, h2, h3 = self.get_methyl_hydrogens(c.N, c.CA, c.C, length=AMINE_LEN)
            self.terminal_atoms.update({"H2": h2, "H3": h3})
            hs.append(h)  # Used as normal amine hydrogen, H
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

        # Terminal atoms
        if n_terminal:
            hs.extend([self.terminal_atoms["H2"], self.terminal_atoms["H3"]])
        if c_terminal:
            oxt = self.get_amide_methine_hydrogen(c.CA, c.C, c.O, oxt=True)
            self.terminal_atoms.update({"OXT": oxt})
            hs.append(oxt)

        hs = self.pad_hydrogens(resname, hs)
        if self.device == 'cuda':
            hs = [h.cuda() for h in hs]

        return self.stack(hs, 0)


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


###########################################
# Torchscript and numba helper functions. #
###########################################


@njit
def _M(axis: np.ndarray, theta):
    """Numba compiled function for generating rotation matrix. See HB.M()"""
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    r = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                  [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                  [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

    return r


@njit
def _M_posneg(axis: np.ndarray, theta):
    """Numba compiled function for generating two similar rotation matrices."""
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    diag0 = aa + bb - cc - dd
    diag1 = a + cc - bb - dd
    diag2 = aa + dd - bb - cc
    r1 = np.array([[diag0, 2 * (bc + ad), 2 * (bd - ac)],
                  [2 * (bc - ad), diag1, 2 * (cd + ab)],
                  [2 * (bd + ac), 2 * (cd - ab), diag2]])
    r2 = np.array([[diag0, 2 * (bc - ad), 2 * (bd + ac)],
                   [2 * (bc + ad), diag1, 2 * (cd - ab)],
                   [2 * (bd - ac), 2 * (cd + ab), diag2]])

    return r1, r2


# @torch.jit.script
def _Mt(axis, theta):
    """Torchscript compiled version of HB.M() for generating a rotation matrix."""
    axis = axis / torch.sqrt(torch.dot(axis, axis))
    a = torch.cos(theta / 2.0)
    intermediate = -axis * torch.sin(theta / 2.0)
    b, c, d = intermediate[0], intermediate[1], intermediate[2]
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    r = torch.zeros((3, 3))
    r[0, 0] = aa + bb - cc - dd
    r[0, 1] = 2 * (bc + ad)
    r[0, 2] = 2 * (bd - ac)
    r[1, 0] = 2 * (bc - ad)
    r[1, 1] = aa + cc - bb - dd
    r[1, 2] = 2 * (cd + ab)
    r[2, 0] = 2 * (bd + ac)
    r[2, 1] = 2 * (cd - ab)
    r[2, 2] = aa + dd - bb - cc
    return r.double()


# @torch.jit.script
def _scale(vector, target_len):
    """Torchscript version of HydrogenBuilder._scale()."""
    v_len = torch.norm(vector)
    return vector / v_len * target_len


# @torch.jit.script
def _scale_l(vector, target_len, v_len):
    """Torchscript version of HydrogenBuilder._scale() with length provided."""
    return vector / v_len * target_len


# @torch.jit.script
def _get_methyl_hydrogens(carbon, prev1, prev2, length, met_angle, rad120):
    """Place methyl (H3) hydrogens on a Carbon atom. Also supports N-terminal amines.

    Ex: Alanine: carbon, prev1, prev2 are CB, CA, N.
    """
    # Define local vectors extending from CA
    N = prev2 - prev1
    CB = carbon - prev1

    # Define perpendicular vector
    PV = torch.cross(CB, N)
    R109 = _Mt(PV, met_angle)  # Rotate abt PV by 109.5 (tetrahed.)

    # Define Hydrogen extending from carbon
    H1 = torch.matmul(R109, -CB)  # Place Hydrogen by rotating C along perpendicular axis
    H1 = _scale(H1, length) + carbon

    R120 = _Mt(CB, rad120)
    H2 = torch.matmul(R120, H1.clone())  # Place 2nd Hydrogen by rotating prev H 120 deg
    H3 = torch.matmul(R120, H2.clone())  # Place 3rd Hydrogen by rotating prev H 120 deg

    return [H1, H2, H3]


# @torch.jit.script
def _get_single_sp3_hydrogen_help(center, R1, R2, R3):
    """Torchscript helper for _get_single_sp3_hydrogen."""
    return -R1 - R2 - R3 + (3 * center)


# @torch.jit.script
def _get_amide_methine_hydrogen_help(R1, center, R2, length):
    """Torchscript helper for _get_amide_methine_hydrogen."""
    return _scale(-R1 - R2 + 2*center, length) + center


ATOM_MAP_H = {}
for one_letter, three_letter in ONE_TO_THREE_LETTER_MAP.items():
    ATOM_MAP_H[one_letter] = ["N", "CA", "C", "O"] + list(
        SC_BUILD_INFO[ONE_TO_THREE_LETTER_MAP[one_letter]]["atom-names"])
    ATOM_MAP_H[one_letter].extend(HYDROGEN_NAMES[three_letter])
    ATOM_MAP_H[one_letter].extend(
        ["PAD"] * (NUM_COORDS_PER_RES_W_HYDROGENS - len(ATOM_MAP_H[one_letter])))
