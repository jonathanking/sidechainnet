"""Hard-coded values to use when building and analyzing sidechain structures.

Bond lengths and angles are programmatically parsed from AMBER forcefields.
'p' and 'i' for 'torsion-vals' stand for predicted and inferred, respectively.

SIDECHAIN DATA FORMAT

    Angle vectors (NUM_ANGLES) are:
        [phi, psi, omega, n-ca-c, ca-c-n, c-n-ca, x0, x1, x2, x3, x4, x5]
        [   bb torsion  |    bb 3-atom angles   |   sidechain torsion   ]

        Notes:
            - x0 is defined as the torsional angle used to place CB using
            (Ci-1, N, CA, CB) or (Ni+1, Ci+1, CA, CB) depending on whether or not the
            previous or following residue is available for measurement.
            - x5 is used to place NH1 or NH2 of Arginine.
            - if a given angle is not present, it is recorded with a GLOBAL_PAD_CHAR.

    Coordinate vectors (NUM_COORDS_PER_RES x 3) for resname RES are:
        [N, CA, C, O, *SC_BUILD_INFO[RES]['atom_names'], <PAD> * (N_PAD)]
        [ backbone  |          sidechain atoms         |     padding*   ]
        where each vector is padded with GLOBAL_PAD_CHAR to maximum length.

        for example, the atoms for an ASN residue are:
            [N, CA, C, O, CB, CG, OD1, ND2, PAD, PAD, PAD, PAD, PAD, PAD]
"""

NUM_ANGLES = 12
NUM_BB_TORSION_ANGLES = 3
NUM_BB_OTHER_ANGLES = 3
NUM_SC_ANGLES = NUM_ANGLES - (NUM_BB_OTHER_ANGLES + NUM_BB_TORSION_ANGLES)
SC_ANGLES_START_POS = NUM_BB_OTHER_ANGLES + NUM_BB_TORSION_ANGLES

NUM_COORDS_PER_RES = 14
PRODY_CA_DIST = 4.1

ANGLE_NAME_TO_IDX_MAP = {
    # see notes for x0 and x5
    'phi': 0,
    'psi': 1,
    'omega': 2,
    'n-ca-c': 3,
    'ca-c-n': 4,
    'c-n-ca': 5,
    'x0': 6,
    'x1': 7,
    'x2': 8,
    'x3': 9,
    'x4': 10,
    'x5': 11
}

ANGLE_IDX_TO_NAME_MAP = {idx: name for name, idx in ANGLE_NAME_TO_IDX_MAP.items()}

SC_BUILD_INFO = {
    'ALA': {
        'angles-names': ['N-CA-CB'],
        'angles-types': ['N -CX-CT'],
        'angles-vals': [1.9146261894377796],
        'atom-names': ['CB'],
        'bonds-names': ['CA-CB'],
        'bonds-types': ['CX-CT'],
        'bonds-vals': [1.526],
        'torsion-names': ['C-N-CA-CB'],
        'torsion-types': ['C -N -CX-CT'],
        'torsion-vals': ['p']
    },
    'ARG': {
        'angles-names': [
            'N-CA-CB', 'CA-CB-CG', 'CB-CG-CD', 'CG-CD-NE', 'CD-NE-CZ', 'NE-CZ-NH1',
            'NE-CZ-NH2'
        ],
        'angles-types': [
            'N -CX-C8', 'CX-C8-C8', 'C8-C8-C8', 'C8-C8-N2', 'C8-N2-CA', 'N2-CA-N2',
            'N2-CA-N2'
        ],
        'angles-vals': [
            1.9146261894377796, 1.911135530933791, 1.911135530933791, 1.9408061282176945,
            2.150245638457014, 2.0943951023931953, 2.0943951023931953
        ],
        'atom-names': ['CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2'],
        'bonds-names': ['CA-CB', 'CB-CG', 'CG-CD', 'CD-NE', 'NE-CZ', 'CZ-NH1', 'CZ-NH2'],
        'bonds-types': ['CX-C8', 'C8-C8', 'C8-C8', 'C8-N2', 'N2-CA', 'CA-N2', 'CA-N2'],
        'bonds-vals': [1.526, 1.526, 1.526, 1.463, 1.34, 1.34, 1.34],
        'torsion-names': [
            'C-N-CA-CB', 'N-CA-CB-CG', 'CA-CB-CG-CD', 'CB-CG-CD-NE', 'CG-CD-NE-CZ',
            'CD-NE-CZ-NH1', 'CD-NE-CZ-NH2'
        ],
        'torsion-types': [
            'C -N -CX-C8', 'N -CX-C8-C8', 'CX-C8-C8-C8', 'C8-C8-C8-N2', 'C8-C8-N2-CA',
            'C8-N2-CA-N2', 'C8-N2-CA-N2'
        ],
        'torsion-vals': ['p', 'p', 'p', 'p', 'p', 'p', 'i']
    },
    'ASN': {
        'angles-names': ['N-CA-CB', 'CA-CB-CG', 'CB-CG-OD1', 'CB-CG-ND2'],
        'angles-types': ['N -CX-2C', 'CX-2C-C ', '2C-C -O ', '2C-C -N '],
        'angles-vals': [
            1.9146261894377796, 1.9390607989657, 2.101376419401173, 2.035053907825388
        ],
        'atom-names': ['CB', 'CG', 'OD1', 'ND2'],
        'bonds-names': ['CA-CB', 'CB-CG', 'CG-OD1', 'CG-ND2'],
        'bonds-types': ['CX-2C', '2C-C ', 'C -O ', 'C -N '],
        'bonds-vals': [1.526, 1.522, 1.229, 1.335],
        'torsion-names': ['C-N-CA-CB', 'N-CA-CB-CG', 'CA-CB-CG-OD1', 'CA-CB-CG-ND2'],
        'torsion-types': ['C -N -CX-2C', 'N -CX-2C-C ', 'CX-2C-C -O ', 'CX-2C-C -N '],
        'torsion-vals': ['p', 'p', 'p', 'i']
    },
    'ASP': {
        'angles-names': ['N-CA-CB', 'CA-CB-CG', 'CB-CG-OD1', 'CB-CG-OD2'],
        'angles-types': ['N -CX-2C', 'CX-2C-CO', '2C-CO-O2', '2C-CO-O2'],
        'angles-vals': [
            1.9146261894377796, 1.9390607989657, 2.0420352248333655, 2.0420352248333655
        ],
        'atom-names': ['CB', 'CG', 'OD1', 'OD2'],
        'bonds-names': ['CA-CB', 'CB-CG', 'CG-OD1', 'CG-OD2'],
        'bonds-types': ['CX-2C', '2C-CO', 'CO-O2', 'CO-O2'],
        'bonds-vals': [1.526, 1.522, 1.25, 1.25],
        'torsion-names': ['C-N-CA-CB', 'N-CA-CB-CG', 'CA-CB-CG-OD1', 'CA-CB-CG-OD2'],
        'torsion-types': ['C -N -CX-2C', 'N -CX-2C-CO', 'CX-2C-CO-O2', 'CX-2C-CO-O2'],
        'torsion-vals': ['p', 'p', 'p', 'i']
    },
    'CYS': {
        'angles-names': ['N-CA-CB', 'CA-CB-SG'],
        'angles-types': ['N -CX-2C', 'CX-2C-SH'],
        'angles-vals': [1.9146261894377796, 1.8954275676658419],
        'atom-names': ['CB', 'SG'],
        'bonds-names': ['CA-CB', 'CB-SG'],
        'bonds-types': ['CX-2C', '2C-SH'],
        'bonds-vals': [1.526, 1.81],
        'torsion-names': ['C-N-CA-CB', 'N-CA-CB-SG'],
        'torsion-types': ['C -N -CX-2C', 'N -CX-2C-SH'],
        'torsion-vals': ['p', 'p']
    },
    'GLN': {
        'angles-names': ['N-CA-CB', 'CA-CB-CG', 'CB-CG-CD', 'CG-CD-OE1', 'CG-CD-NE2'],
        'angles-types': ['N -CX-2C', 'CX-2C-2C', '2C-2C-C ', '2C-C -O ', '2C-C -N '],
        'angles-vals': [
            1.9146261894377796, 1.911135530933791, 1.9390607989657, 2.101376419401173,
            2.035053907825388
        ],
        'atom-names': ['CB', 'CG', 'CD', 'OE1', 'NE2'],
        'bonds-names': ['CA-CB', 'CB-CG', 'CG-CD', 'CD-OE1', 'CD-NE2'],
        'bonds-types': ['CX-2C', '2C-2C', '2C-C ', 'C -O ', 'C -N '],
        'bonds-vals': [1.526, 1.526, 1.522, 1.229, 1.335],
        'torsion-names': [
            'C-N-CA-CB', 'N-CA-CB-CG', 'CA-CB-CG-CD', 'CB-CG-CD-OE1', 'CB-CG-CD-NE2'
        ],
        'torsion-types': [
            'C -N -CX-2C', 'N -CX-2C-2C', 'CX-2C-2C-C ', '2C-2C-C -O ', '2C-2C-C -N '
        ],
        'torsion-vals': ['p', 'p', 'p', 'p', 'i']
    },
    'GLU': {
        'angles-names': ['N-CA-CB', 'CA-CB-CG', 'CB-CG-CD', 'CG-CD-OE1', 'CG-CD-OE2'],
        'angles-types': ['N -CX-2C', 'CX-2C-2C', '2C-2C-CO', '2C-CO-O2', '2C-CO-O2'],
        'angles-vals': [
            1.9146261894377796, 1.911135530933791, 1.9390607989657, 2.0420352248333655,
            2.0420352248333655
        ],
        'atom-names': ['CB', 'CG', 'CD', 'OE1', 'OE2'],
        'bonds-names': ['CA-CB', 'CB-CG', 'CG-CD', 'CD-OE1', 'CD-OE2'],
        'bonds-types': ['CX-2C', '2C-2C', '2C-CO', 'CO-O2', 'CO-O2'],
        'bonds-vals': [1.526, 1.526, 1.522, 1.25, 1.25],
        'torsion-names': [
            'C-N-CA-CB', 'N-CA-CB-CG', 'CA-CB-CG-CD', 'CB-CG-CD-OE1', 'CB-CG-CD-OE2'
        ],
        'torsion-types': [
            'C -N -CX-2C', 'N -CX-2C-2C', 'CX-2C-2C-CO', '2C-2C-CO-O2', '2C-2C-CO-O2'
        ],
        'torsion-vals': ['p', 'p', 'p', 'p', 'i']
    },
    'GLY': {
        'angles-names': [],
        'angles-types': [],
        'angles-vals': [],
        'atom-names': [],
        'bonds-names': [],
        'bonds-types': [],
        'bonds-vals': [],
        'torsion-names': [],
        'torsion-types': [],
        'torsion-vals': []
    },
    'HIS': {
        'angles-names': [
            'N-CA-CB', 'CA-CB-CG', 'CB-CG-ND1', 'CG-ND1-CE1', 'ND1-CE1-NE2', 'CE1-NE2-CD2'
        ],
        'angles-types': [
            'N -CX-CT', 'CX-CT-CC', 'CT-CC-NA', 'CC-NA-CR', 'NA-CR-NB', 'CR-NB-CV'
        ],
        'angles-vals': [
            1.9146261894377796, 1.9739673840055867, 2.0943951023931953,
            1.8849555921538759, 1.8849555921538759, 1.8849555921538759
        ],
        'atom-names': ['CB', 'CG', 'ND1', 'CE1', 'NE2', 'CD2'],
        'bonds-names': ['CA-CB', 'CB-CG', 'CG-ND1', 'ND1-CE1', 'CE1-NE2', 'NE2-CD2'],
        'bonds-types': ['CX-CT', 'CT-CC', 'CC-NA', 'NA-CR', 'CR-NB', 'NB-CV'],
        'bonds-vals': [1.526, 1.504, 1.385, 1.343, 1.335, 1.394],
        'torsion-names': [
            'C-N-CA-CB', 'N-CA-CB-CG', 'CA-CB-CG-ND1', 'CB-CG-ND1-CE1', 'CG-ND1-CE1-NE2',
            'ND1-CE1-NE2-CD2'
        ],
        'torsion-types': [
            'C -N -CX-CT', 'N -CX-CT-CC', 'CX-CT-CC-NA', 'CT-CC-NA-CR', 'CC-NA-CR-NB',
            'NA-CR-NB-CV'
        ],
        'torsion-vals': ['p', 'p', 'p', 3.141592653589793, 0.0, 0.0]
    },
    'ILE': {
        'angles-names': ['N-CA-CB', 'CA-CB-CG1', 'CB-CG1-CD1', 'CA-CB-CG2'],
        'angles-types': ['N -CX-3C', 'CX-3C-2C', '3C-2C-CT', 'CX-3C-CT'],
        'angles-vals': [
            1.9146261894377796, 1.911135530933791, 1.911135530933791, 1.911135530933791
        ],
        'atom-names': ['CB', 'CG1', 'CD1', 'CG2'],
        'bonds-names': ['CA-CB', 'CB-CG1', 'CG1-CD1', 'CB-CG2'],
        'bonds-types': ['CX-3C', '3C-2C', '2C-CT', '3C-CT'],
        'bonds-vals': [1.526, 1.526, 1.526, 1.526],
        'torsion-names': ['C-N-CA-CB', 'N-CA-CB-CG1', 'CA-CB-CG1-CD1', 'N-CA-CB-CG2'],
        'torsion-types': ['C -N -CX-3C', 'N -CX-3C-2C', 'CX-3C-2C-CT', 'N -CX-3C-CT'],
        'torsion-vals': ['p', 'p', 'p', 'p']
    },
    'LEU': {
        'angles-names': ['N-CA-CB', 'CA-CB-CG', 'CB-CG-CD1', 'CB-CG-CD2'],
        'angles-types': ['N -CX-2C', 'CX-2C-3C', '2C-3C-CT', '2C-3C-CT'],
        'angles-vals': [
            1.9146261894377796, 1.911135530933791, 1.911135530933791, 1.911135530933791
        ],
        'atom-names': ['CB', 'CG', 'CD1', 'CD2'],
        'bonds-names': ['CA-CB', 'CB-CG', 'CG-CD1', 'CG-CD2'],
        'bonds-types': ['CX-2C', '2C-3C', '3C-CT', '3C-CT'],
        'bonds-vals': [1.526, 1.526, 1.526, 1.526],
        'torsion-names': ['C-N-CA-CB', 'N-CA-CB-CG', 'CA-CB-CG-CD1', 'CA-CB-CG-CD2'],
        'torsion-types': ['C -N -CX-2C', 'N -CX-2C-3C', 'CX-2C-3C-CT', 'CX-2C-3C-CT'],
        'torsion-vals': ['p', 'p', 'p', 'p']
    },
    'LYS': {
        'angles-names': ['N-CA-CB', 'CA-CB-CG', 'CB-CG-CD', 'CG-CD-CE', 'CD-CE-NZ'],
        'angles-types': ['N -CX-C8', 'CX-C8-C8', 'C8-C8-C8', 'C8-C8-C8', 'C8-C8-N3'],
        'angles-vals': [
            1.9146261894377796, 1.911135530933791, 1.911135530933791, 1.911135530933791,
            1.9408061282176945
        ],
        'atom-names': ['CB', 'CG', 'CD', 'CE', 'NZ'],
        'bonds-names': ['CA-CB', 'CB-CG', 'CG-CD', 'CD-CE', 'CE-NZ'],
        'bonds-types': ['CX-C8', 'C8-C8', 'C8-C8', 'C8-C8', 'C8-N3'],
        'bonds-vals': [1.526, 1.526, 1.526, 1.526, 1.471],
        'torsion-names': [
            'C-N-CA-CB', 'N-CA-CB-CG', 'CA-CB-CG-CD', 'CB-CG-CD-CE', 'CG-CD-CE-NZ'
        ],
        'torsion-types': [
            'C -N -CX-C8', 'N -CX-C8-C8', 'CX-C8-C8-C8', 'C8-C8-C8-C8', 'C8-C8-C8-N3'
        ],
        'torsion-vals': ['p', 'p', 'p', 'p', 'p']
    },
    'MET': {
        'angles-names': ['N-CA-CB', 'CA-CB-CG', 'CB-CG-SD', 'CG-SD-CE'],
        'angles-types': ['N -CX-2C', 'CX-2C-2C', '2C-2C-S ', '2C-S -CT'],
        'angles-vals': [
            1.9146261894377796, 1.911135530933791, 2.0018926520374962, 1.726130630222392
        ],
        'atom-names': ['CB', 'CG', 'SD', 'CE'],
        'bonds-names': ['CA-CB', 'CB-CG', 'CG-SD', 'SD-CE'],
        'bonds-types': ['CX-2C', '2C-2C', '2C-S ', 'S -CT'],
        'bonds-vals': [1.526, 1.526, 1.81, 1.81],
        'torsion-names': ['C-N-CA-CB', 'N-CA-CB-CG', 'CA-CB-CG-SD', 'CB-CG-SD-CE'],
        'torsion-types': ['C -N -CX-2C', 'N -CX-2C-2C', 'CX-2C-2C-S ', '2C-2C-S -CT'],
        'torsion-vals': ['p', 'p', 'p', 'p']
    },
    'PHE': {
        'angles-names': [
            'N-CA-CB', 'CA-CB-CG', 'CB-CG-CD1', 'CG-CD1-CE1', 'CD1-CE1-CZ', 'CE1-CZ-CE2',
            'CZ-CE2-CD2'
        ],
        'angles-types': [
            'N -CX-CT', 'CX-CT-CA', 'CT-CA-CA', 'CA-CA-CA', 'CA-CA-CA', 'CA-CA-CA',
            'CA-CA-CA'
        ],
        'angles-vals': [
            1.9146261894377796, 1.9896753472735358, 2.0943951023931953,
            2.0943951023931953, 2.0943951023931953, 2.0943951023931953, 2.0943951023931953
        ],
        'atom-names': ['CB', 'CG', 'CD1', 'CE1', 'CZ', 'CE2', 'CD2'],
        'bonds-names': [
            'CA-CB', 'CB-CG', 'CG-CD1', 'CD1-CE1', 'CE1-CZ', 'CZ-CE2', 'CE2-CD2'
        ],
        'bonds-types': ['CX-CT', 'CT-CA', 'CA-CA', 'CA-CA', 'CA-CA', 'CA-CA', 'CA-CA'],
        'bonds-vals': [1.526, 1.51, 1.4, 1.4, 1.4, 1.4, 1.4],
        'torsion-names': [
            'C-N-CA-CB', 'N-CA-CB-CG', 'CA-CB-CG-CD1', 'CB-CG-CD1-CE1', 'CG-CD1-CE1-CZ',
            'CD1-CE1-CZ-CE2', 'CE1-CZ-CE2-CD2'
        ],
        'torsion-types': [
            'C -N -CX-CT', 'N -CX-CT-CA', 'CX-CT-CA-CA', 'CT-CA-CA-CA', 'CA-CA-CA-CA',
            'CA-CA-CA-CA', 'CA-CA-CA-CA'
        ],
        'torsion-vals': ['p', 'p', 'p', 3.141592653589793, 0.0, 0.0, 0.0]
    },
    'PRO': {
        'angles-names': ['N-CA-CB', 'CA-CB-CG', 'CB-CG-CD'],
        'angles-types': ['N -CX-CT', 'CX-CT-CT', 'CT-CT-CT'],
        'angles-vals': [1.9146261894377796, 1.911135530933791, 1.911135530933791],
        'atom-names': ['CB', 'CG', 'CD'],
        'bonds-names': ['CA-CB', 'CB-CG', 'CG-CD'],
        'bonds-types': ['CX-CT', 'CT-CT', 'CT-CT'],
        'bonds-vals': [1.526, 1.526, 1.526],
        'torsion-names': ['C-N-CA-CB', 'N-CA-CB-CG', 'CA-CB-CG-CD'],
        'torsion-types': ['C -N -CX-CT', 'N -CX-CT-CT', 'CX-CT-CT-CT'],
        'torsion-vals': ['p', 'p', 'p']
    },
    'SER': {
        'angles-names': ['N-CA-CB', 'CA-CB-OG'],
        'angles-types': ['N -CX-2C', 'CX-2C-OH'],
        'angles-vals': [1.9146261894377796, 1.911135530933791],
        'atom-names': ['CB', 'OG'],
        'bonds-names': ['CA-CB', 'CB-OG'],
        'bonds-types': ['CX-2C', '2C-OH'],
        'bonds-vals': [1.526, 1.41],
        'torsion-names': ['C-N-CA-CB', 'N-CA-CB-OG'],
        'torsion-types': ['C -N -CX-2C', 'N -CX-2C-OH'],
        'torsion-vals': ['p', 'p']
    },
    'THR': {
        'angles-names': ['N-CA-CB', 'CA-CB-OG1', 'CA-CB-CG2'],
        'angles-types': ['N -CX-3C', 'CX-3C-OH', 'CX-3C-CT'],
        'angles-vals': [1.9146261894377796, 1.911135530933791, 1.911135530933791],
        'atom-names': ['CB', 'OG1', 'CG2'],
        'bonds-names': ['CA-CB', 'CB-OG1', 'CB-CG2'],
        'bonds-types': ['CX-3C', '3C-OH', '3C-CT'],
        'bonds-vals': [1.526, 1.41, 1.526],
        'torsion-names': ['C-N-CA-CB', 'N-CA-CB-OG1', 'N-CA-CB-CG2'],
        'torsion-types': ['C -N -CX-3C', 'N -CX-3C-OH', 'N -CX-3C-CT'],
        'torsion-vals': ['p', 'p', 'p']
    },
    'TRP': {
        'angles-names': [
            'N-CA-CB', 'CA-CB-CG', 'CB-CG-CD1', 'CG-CD1-NE1', 'CD1-NE1-CE2',
            'NE1-CE2-CZ2', 'CE2-CZ2-CH2', 'CZ2-CH2-CZ3', 'CH2-CZ3-CE3', 'CZ3-CE3-CD2'
        ],
        'angles-types': [
            'N -CX-CT', 'CX-CT-C*', 'CT-C*-CW', 'C*-CW-NA', 'CW-NA-CN', 'NA-CN-CA',
            'CN-CA-CA', 'CA-CA-CA', 'CA-CA-CA', 'CA-CA-CB'
        ],
        'angles-vals': [
            1.9146261894377796, 2.0176006153054447, 2.181661564992912, 1.8971728969178363,
            1.9477874452256716, 2.3177972466484698, 2.0943951023931953,
            2.0943951023931953, 2.0943951023931953, 2.0943951023931953
        ],
        'atom-names': [
            'CB', 'CG', 'CD1', 'NE1', 'CE2', 'CZ2', 'CH2', 'CZ3', 'CE3', 'CD2'
        ],
        'bonds-names': [
            'CA-CB', 'CB-CG', 'CG-CD1', 'CD1-NE1', 'NE1-CE2', 'CE2-CZ2', 'CZ2-CH2',
            'CH2-CZ3', 'CZ3-CE3', 'CE3-CD2'
        ],
        'bonds-types': [
            'CX-CT', 'CT-C*', 'C*-CW', 'CW-NA', 'NA-CN', 'CN-CA', 'CA-CA', 'CA-CA',
            'CA-CA', 'CA-CB'
        ],
        'bonds-vals': [1.526, 1.495, 1.352, 1.381, 1.38, 1.4, 1.4, 1.4, 1.4, 1.404],
        'torsion-names': [
            'C-N-CA-CB', 'N-CA-CB-CG', 'CA-CB-CG-CD1', 'CB-CG-CD1-NE1', 'CG-CD1-NE1-CE2',
            'CD1-NE1-CE2-CZ2', 'NE1-CE2-CZ2-CH2', 'CE2-CZ2-CH2-CZ3', 'CZ2-CH2-CZ3-CE3',
            'CH2-CZ3-CE3-CD2'
        ],
        'torsion-types': [
            'C -N -CX-CT', 'N -CX-CT-C*', 'CX-CT-C*-CW', 'CT-C*-CW-NA', 'C*-CW-NA-CN',
            'CW-NA-CN-CA', 'NA-CN-CA-CA', 'CN-CA-CA-CA', 'CA-CA-CA-CA', 'CA-CA-CA-CB'
        ],
        'torsion-vals': [
            'p', 'p', 'p', 3.141592653589793, 0.0, 3.141592653589793, 3.141592653589793,
            0.0, 0.0, 0.0
        ]
    },
    'TYR': {
        'angles-names': [
            'N-CA-CB', 'CA-CB-CG', 'CB-CG-CD1', 'CG-CD1-CE1', 'CD1-CE1-CZ', 'CE1-CZ-OH',
            'CE1-CZ-CE2', 'CZ-CE2-CD2'
        ],
        'angles-types': [
            'N -CX-CT', 'CX-CT-CA', 'CT-CA-CA', 'CA-CA-CA', 'CA-CA-C ', 'CA-C -OH',
            'CA-C -CA', 'C -CA-CA'
        ],
        'angles-vals': [
            1.9146261894377796, 1.9896753472735358, 2.0943951023931953,
            2.0943951023931953, 2.0943951023931953, 2.0943951023931953,
            2.0943951023931953, 2.0943951023931953
        ],
        'atom-names': ['CB', 'CG', 'CD1', 'CE1', 'CZ', 'OH', 'CE2', 'CD2'],
        'bonds-names': [
            'CA-CB', 'CB-CG', 'CG-CD1', 'CD1-CE1', 'CE1-CZ', 'CZ-OH', 'CZ-CE2', 'CE2-CD2'
        ],
        'bonds-types': [
            'CX-CT', 'CT-CA', 'CA-CA', 'CA-CA', 'CA-C ', 'C -OH', 'C -CA', 'CA-CA'
        ],
        'bonds-vals': [1.526, 1.51, 1.4, 1.4, 1.409, 1.364, 1.409, 1.4],
        'torsion-names': [
            'C-N-CA-CB', 'N-CA-CB-CG', 'CA-CB-CG-CD1', 'CB-CG-CD1-CE1', 'CG-CD1-CE1-CZ',
            'CD1-CE1-CZ-OH', 'CD1-CE1-CZ-CE2', 'CE1-CZ-CE2-CD2'
        ],
        'torsion-types': [
            'C -N -CX-CT', 'N -CX-CT-CA', 'CX-CT-CA-CA', 'CT-CA-CA-CA', 'CA-CA-CA-C ',
            'CA-CA-C -OH', 'CA-CA-C -CA', 'CA-C -CA-CA'
        ],
        'torsion-vals': [
            'p', 'p', 'p', 3.141592653589793, 0.0, 3.141592653589793, 0.0, 0.0
        ]
    },
    'VAL': {
        'angles-names': ['N-CA-CB', 'CA-CB-CG1', 'CA-CB-CG2'],
        'angles-types': ['N -CX-3C', 'CX-3C-CT', 'CX-3C-CT'],
        'angles-vals': [1.9146261894377796, 1.911135530933791, 1.911135530933791],
        'atom-names': ['CB', 'CG1', 'CG2'],
        'bonds-names': ['CA-CB', 'CB-CG1', 'CB-CG2'],
        'bonds-types': ['CX-3C', '3C-CT', '3C-CT'],
        'bonds-vals': [1.526, 1.526, 1.526],
        'torsion-names': ['C-N-CA-CB', 'N-CA-CB-CG1', 'N-CA-CB-CG2'],
        'torsion-types': ['C -N -CX-3C', 'N -CX-3C-CT', 'N -CX-3C-CT'],
        'torsion-vals': ['p', 'p', 'p']
    }
}

BB_BUILD_INFO = {
    "BONDLENS": {
        'n-ca': 1.442,
        'ca-c': 1.498,
        'c-n': 1.379,
        'c-o': 1.229,  # From parm10.dat
        'c-oh': 1.364
    },  # From parm10.dat, for OXT
    # For placing oxygens
    "BONDANGS": {
        'ca-c-o': 2.0944,  # Approximated to be 2pi / 3; parm10.dat says 2.0350539
        'ca-c-oh': 2.0944
    },  # Equal to 'ca-c-o', for OXT
    "BONDTORSIONS": {
        'n-ca-c-n': -0.785398163
    }  # A simple approximation, not meant to be exact.
}
