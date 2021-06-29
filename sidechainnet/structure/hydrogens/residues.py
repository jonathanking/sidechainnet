"""Methods for adding hydrogen atoms to amino acid residues."""
from sidechainnet.structure.PdbBuilder import ATOM_MAP_14
from sidechainnet.utils.measure import GLOBAL_PAD_CHAR
from . import hydrogens as hy
import numpy as np


def pad_hydrogens(resname, hydrogens):
    """Pad a hydrogen list with empty vectors to the correct length for a given res."""
    pad_vec = [GLOBAL_PAD_CHAR * np.ones(3)]
    n_heavy_atoms = sum([True if an != "PAD" else False for an in ATOM_MAP_14[resname]])
    n_pad = hy.NUM_COORDS_PER_RES_W_HYDROGENS - n_heavy_atoms - len(hydrogens)
    hydrogens.extend(pad_vec * (n_pad))
    return hydrogens


def get_hydrogens_for_res(resname, c, prevc):
    """Return a padded list of hydrogens for a given residue name and atom coord tuple."""
    # All amino acids have an amide-hydrogen along the backbone; terminal NH3 not yet supported
    hs = []
    if prevc:
        hs.append(hy.get_amide_methine_hydrogen(prevc.C, c.N, c.CA, amide=True))
    # If the amino acid is not Glycine, we can add an sp3-hybridized H to CA
    if resname != "G":
        cah = hy.get_single_sp3_hydrogen(center=c.CA, R1=c.N, R2=c.C, R3=c.CB)
        hs.append(cah)
    # Now, we can add the remaining unique hydrogens for each amino acid
    if resname == "A":
        hs.extend(ala(c))
    elif resname == "R":
        hs.extend(arg(c))
    elif resname == "N":
        hs.extend(asn(c))
    elif resname == "D":
        hs.extend(asp(c))
    elif resname == "C":
        hs.extend(cys(c))
    elif resname == "E":
        hs.extend(glu(c))
    elif resname == "Q":
        hs.extend(gln(c))
    elif resname == "G":
        hs.extend(gly(c))
    elif resname == "H":
        hs.extend(his(c))
    elif resname == "I":
        hs.extend(ile(c))
    elif resname == "L":
        hs.extend(leu(c))
    elif resname == "K":
        hs.extend(lys(c))
    elif resname == "M":
        hs.extend(met(c))
    elif resname == "F":
        hs.extend(phe(c))
    elif resname == "P":
        hs.extend(pro(c))
    elif resname == "S":
        hs.extend(ser(c))
    elif resname == "T":
        hs.extend(thr(c))
    elif resname == "W":
        hs.extend(trp(c))
    elif resname == "Y":
        hs.extend(tyr(c))
    elif resname == "V":
        hs.extend(val(c))

    return pad_hydrogens(resname, hs)


def ala(c):
    """Return list of hydrogen positions for ala.

    Positions correspond to the following hydrogens:
        [HB1, HB2, HB3]
    """
    hs = hy.get_methyl_hydrogens(carbon=c.CB, prev1=c.CA, prev2=c.N)  # HB1, HB2, HB3
    return hs


def arg(c):
    """Return list of hydrogen positions for arg.

    Positions correspond to the following hydrogens:
        [HB2, HB3, HD2, HD3, HG2, HG3, HE, HH11, HH12, HH21, HH22]
    """
    hs = []
    # Methylene hydrogens: [HB2, HB3, HD2, HD3, HG2, HG3]
    for R1, carbon, R2 in [(c.CA, c.CB, c.CG), (c.CB, c.CG, c.CD), (c.CG, c.CD, c.NE)]:
        hs.extend(hy.get_methylene_hydrogens(R1, carbon, R2))
    # Amide hydrogen: HE
    hs.append(hy.get_amide_methine_hydrogen(c.CD, c.NE, c.CZ, amide=True))
    # Amide hydrogens: [HH11, HH12, HH21, HH22]
    hs.extend(hy.get_amine_hydrogens(c.NH1, c.CZ, c.NE))
    hs.extend(hy.get_amine_hydrogens(c.NH2, c.CZ, c.NE))

    return hs


def asn(c):
    """Return list of hydrogen positions for asn.

    Positions correspond to the following hydrogens:
        [HB2, HB3, HD21, HD22]
    """
    pass


def asp(c):
    """Return list of hydrogen positions for asp.

    Positions correspond to the following hydrogens:
        [HB2, HB3]
    """
    pass


def cys(c):
    """Return list of hydrogen positions for cys.

    Positions correspond to the following hydrogens:
        [HB2, HB3, HG]
    """
    pass


def gln(c):
    """Return list of hydrogen positions for gln.

    Positions correspond to the following hydrogens:
        [HB2, HB3, HE21, HE22, HG2, HG3]
    """
    pass


def glu(c):
    """Return list of hydrogen positions for glu.

    Positions correspond to the following hydrogens:
        [HB2, HB3, HG2, HG3]
    """
    pass


def gly(c):
    """Return list of hydrogen positions for gly.

    Positions correspond to the following hydrogens:
        [H, HA2, HA3]
    """
    pass


def his(c):
    """Return list of hydrogen positions for his.

    Positions correspond to the following hydrogens:
        [HB2, HB3, HD2, HE1]
    """
    pass


def ile(c):
    """Return list of hydrogen positions for ile.

    Positions correspond to the following hydrogens:
        [HB, HD11, HD12, HD13, HG12, HG13, HG21, HG22, HG23]
    """
    pass


def leu(c):
    """Return list of hydrogen positions for leu.

    Positions correspond to the following hydrogens:
        [HB2, HB3, HD11, HD12, HD13, HD21, HD22, HD23, HG]
    """
    pass


def lys(c):
    """Return list of hydrogen positions for lys.

    Positions correspond to the following hydrogens:
        [HB2, HB3, HD2, HD3, HE2, HE3, HG2, HG3, HZ1, HZ2, HZ3]
    """
    pass


def met(c):
    """Return list of hydrogen positions for met.

    Positions correspond to the following hydrogens:
        [HB2, HB3, HE1, HE2, HE3, HG2, HG3]
    """
    pass


def phe(c):
    """Return list of hydrogen positions for phe.

    Positions correspond to the following hydrogens:
        [HB2, HB3, HD1, HD2, HE1, HE2, HZ]
    """
    pass


def pro(c):
    """Return list of hydrogen positions for pro.

    Positions correspond to the following hydrogens:
        [HA, HB2, HB3, HD2, HD3, HG2, HG3]
    """
    pass


def ser(c):
    """Return list of hydrogen positions for ser.

    Positions correspond to the following hydrogens:
        [HB2, HB3, HG]
    """
    pass


def thr(c):
    """Return list of hydrogen positions for thr.

    Positions correspond to the following hydrogens:
        [HB, HG1, HG21, HG22, HG23]
    """
    pass


def trp(c):
    """Return list of hydrogen positions for trp.

    Positions correspond to the following hydrogens:
        [HB2, HB3, HD1, HE1, HE3, HH2, HZ2, HZ3]
    """
    pass


def tyr(c):
    """Return list of hydrogen positions for tyr.

    Positions correspond to the following hydrogens:
        [HB2, HB3, HD1, HD2, HE1, HE2, HH]
    """
    pass


def val(c):
    """Return list of hydrogen positions for val.

    Positions correspond to the following hydrogens:
        [HB, HG11, HG12, HG13, HG21, HG22, HG23]
    """
    pass

