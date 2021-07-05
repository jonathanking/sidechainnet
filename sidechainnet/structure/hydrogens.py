"""Methods for adding hydrogen atoms to amino acid residues."""
import numpy as np
import torch
from scipy.linalg import expm
from sidechainnet.structure.build_info import SC_BUILD_INFO
from sidechainnet.utils.measure import GLOBAL_PAD_CHAR
from sidechainnet.utils.sequence import ONE_TO_THREE_LETTER_MAP

NUM_COORDS_PER_RES_W_HYDROGENS = 24

METHYL_ANGLE = 109.5
METHYL_LEN = 1.09

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


class HydrogenBuilder(object):

    def __init__(self, seq, coords):
        from sidechainnet.structure.PdbBuilder import ATOM_MAP_14

        self.seq = seq
        self.coords = coords
        self.mode = "numpy"
        self.atom_map = ATOM_MAP_14

        self.norm = np.linalg.norm if self.mode == "numpy" else torch.norm
        self.cross = np.cross if self.mode == "numpy" else torch.cross
        self.dot = np.dot if self.mode == "numpy" else torch.matmul
        self.eye = np.eye if self.mode == "numpy" else torch.eye
        self.ones = np.ones if self.mode == "numpy" else torch.ones

    # Utility functions
    def M(self, axis, theta):
        """Create rotation matrix with angle theta around a given axis.

        From https://stackoverflow.com/questions/6802577/rotation-of-3d-vector.
        """
        # TODO find torch expm fn
        m = expm(self.cross(self.eye(3), axis / self.norm(axis) * theta))
        if self.mode == "numpy":
            return m
        return torch.tensor(m)

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

    def get_methyl_hydrogens(self, carbon, prev1, prev2):
        """Place methyl (H3) hydrogens on a Carbon atom.

        Ex: Alanine: carbon, prev1, prev2 are CB, CA, N.
        """
        # TODO move to class - remove global variable; easier switch; save states
        # Define local vectors extending from CA
        N = prev2 - prev1
        CB = carbon - prev1

        # Define perpendicular vector
        PV = self.cross(CB, N)
        R109 = self.M(PV,
                      np.deg2rad(METHYL_ANGLE))  # Rotate around PV by 109.5 (tetrahedral)

        # Define Hydrogen extending from carbon
        H1 = self.dot(R109, -CB)  # Place Hydrogen by rotating C along perpendicular axis
        H1 = self.scale(H1, METHYL_LEN)

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

    def get_thiol_hydrogen(self, oxy_sulfur, prev1, prev2):
        # Define local vectors
        OS = oxy_sulfur - prev1
        P2 = prev2 - prev1

        # Define perpendicular vector  to other
        PV = self.cross(OS, P2)

        # Define rotation matrices
        RP = self.M(PV, np.pi -
                    np.deg2rad(THIOL_ANGLE))  # Rotate around PV by 109.5 (tetrahedral)
        RQ = self.M(OS, np.pi / 2)  # Rotate around thiol axis by 1/4 turn

        # Define Hydrogens
        H1 = self.dot(RQ, self.dot(RP, OS))  # Place Hydrogen by rotating OS vec twice
        H1 = self.scale(H1, THIOL_LEN)
        H1 += OS

        return H1 + prev1

    def get_amide_methine_hydrogen(self, R1, center, R2, amide=True):
        length = AMIDE_LEN if amide else METHINE_LEN

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
        hs = self.get_methyl_hydrogens(carbon=c.CB, prev1=c.CA,
                                       prev2=c.N)  # HB1, HB2, HB3
        return hs

    def arg(self, c):
        """Return list of hydrogen positions for arg.

        Positions correspond to the following hydrogens:
            [HB2, HB3, HD2, HD3, HG2, HG3, HE, HH11, HH12, HH21, HH22]
        """
        hs = []
        # Methylene hydrogens: [HB2, HB3, HD2, HD3, HG2, HG3]
        for R1, carbon, R2 in [(c.CA, c.CB, c.CG), (c.CB, c.CG, c.CD),
                               (c.CG, c.CD, c.NE)]:
            hs.extend(self.get_methylene_hydrogens(R1, carbon, R2))
        # Amide hydrogen: HE
        hs.append(self.get_amide_methine_hydrogen(c.CD, c.NE, c.CZ, amide=True))
        # Amide hydrogens: [HH11, HH12, HH21, HH22]
        hs.extend(self.get_amine_hydrogens(c.NH1, c.CZ, c.NE))
        hs.extend(self.get_amine_hydrogens(c.NH2, c.CZ, c.NE))

        return hs

    def asn(self, c):
        """Return list of hydrogen positions for asn.

        Positions correspond to the following hydrogens:
            [HB2, HB3, HD21, HD22]
        """
        pass

    def asp(self, c):
        """Return list of hydrogen positions for asp.

        Positions correspond to the following hydrogens:
            [HB2, HB3]
        """
        pass

    def cys(self, c):
        """Return list of hydrogen positions for cys.

        Positions correspond to the following hydrogens:
            [HB2, HB3, HG]
        """
        pass

    def gln(self, c):
        """Return list of hydrogen positions for gln.

        Positions correspond to the following hydrogens:
            [HB2, HB3, HE21, HE22, HG2, HG3]
        """
        pass

    def glu(self, c):
        """Return list of hydrogen positions for glu.

        Positions correspond to the following hydrogens:
            [HB2, HB3, HG2, HG3]
        """
        pass

    def gly(self, c):
        """Return list of hydrogen positions for gly.

        Positions correspond to the following hydrogens:
            [H, HA2, HA3]
        """
        pass

    def his(self, c):
        """Return list of hydrogen positions for his.

        Positions correspond to the following hydrogens:
            [HB2, HB3, HD2, HE1]
        """
        pass

    def ile(self, c):
        """Return list of hydrogen positions for ile.

        Positions correspond to the following hydrogens:
            [HB, HD11, HD12, HD13, HG12, HG13, HG21, HG22, HG23]
        """
        pass

    def leu(self, c):
        """Return list of hydrogen positions for leu.

        Positions correspond to the following hydrogens:
            [HB2, HB3, HD11, HD12, HD13, HD21, HD22, HD23, HG]
        """
        pass

    def lys(self, c):
        """Return list of hydrogen positions for lys.

        Positions correspond to the following hydrogens:
            [HB2, HB3, HD2, HD3, HE2, HE3, HG2, HG3, HZ1, HZ2, HZ3]
        """
        pass

    def met(self, c):
        """Return list of hydrogen positions for met.

        Positions correspond to the following hydrogens:
            [HB2, HB3, HE1, HE2, HE3, HG2, HG3]
        """
        pass

    def phe(self, c):
        """Return list of hydrogen positions for phe.

        Positions correspond to the following hydrogens:
            [HB2, HB3, HD1, HD2, HE1, HE2, HZ]
        """
        pass

    def pro(self, c):
        """Return list of hydrogen positions for pro.

        Positions correspond to the following hydrogens:
            [HA, HB2, HB3, HD2, HD3, HG2, HG3]
        """
        pass

    def ser(self, c):
        """Return list of hydrogen positions for ser.

        Positions correspond to the following hydrogens:
            [HB2, HB3, HG]
        """
        pass

    def thr(self, c):
        """Return list of hydrogen positions for thr.

        Positions correspond to the following hydrogens:
            [HB, HG1, HG21, HG22, HG23]
        """
        pass

    def trp(self, c):
        """Return list of hydrogen positions for trp.

        Positions correspond to the following hydrogens:
            [HB2, HB3, HD1, HE1, HE3, HH2, HZ2, HZ3]
        """
        pass

    def tyr(self, c):
        """Return list of hydrogen positions for tyr.

        Positions correspond to the following hydrogens:
            [HB2, HB3, HD1, HD2, HE1, HE2, HH]
        """
        pass

    def val(self, c):
        """Return list of hydrogen positions for val.

        Positions correspond to the following hydrogens:
            [HB, HG11, HG12, HG13, HG21, HG22, HG23]
        """
        pass

    def get_hydrogens_for_res(self, resname, c, prevc):
        """Return a padded list of hydrogens for a given res name & atom coord tuple."""
        # All amino acids have an amide-hydrogen along the backbone
        # TODO Support terminal NH3 instead of None check
        hs = []
        if prevc:
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

        return self.pad_hydrogens(resname, hs)


ATOM_MAP_24 = {}
for one_letter, three_letter in ONE_TO_THREE_LETTER_MAP.items():
    ATOM_MAP_24[one_letter] = ["N", "CA", "C", "O"] + list(
        SC_BUILD_INFO[ONE_TO_THREE_LETTER_MAP[one_letter]]["atom-names"])
    ATOM_MAP_24[one_letter].extend(HYDROGEN_NAMES[three_letter])
    ATOM_MAP_24[one_letter].extend(["PAD"] * (24 - len(ATOM_MAP_24[one_letter])))
