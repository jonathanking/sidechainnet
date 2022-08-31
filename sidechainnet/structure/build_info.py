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
        [N, CA, C, O, *SC_HBUILD_INFO[RES]['atom_names'], <PAD> * (N_PAD)]
        [ backbone  |          sidechain atoms         |     padding*   ]
        where each vector is padded with GLOBAL_PAD_CHAR to maximum length.

        for example, the atoms for an ASN residue are:
            [N, CA, C, O, CB, CG, OD1, ND2, PAD, PAD, PAD, PAD, PAD, PAD]
"""
import copy
import re
import numpy as np
import pkg_resources


PI = np.pi

NUM_ANGLES = 12
NUM_BB_TORSION_ANGLES = 3
NUM_BB_OTHER_ANGLES = 3
NUM_SC_ANGLES = NUM_ANGLES - (NUM_BB_OTHER_ANGLES + NUM_BB_TORSION_ANGLES)
SC_ANGLES_START_POS = NUM_BB_OTHER_ANGLES + NUM_BB_TORSION_ANGLES

NUM_COORDS_PER_RES = 14
PRODY_CA_DIST = 4.1
GLOBAL_PAD_CHAR = np.nan
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
# yapf: disable
_RAW_BUILD_INFO = {
    'ALA': {
        'torsion-names': [
            'C-N-CA-CB',
            # Hydrogens
            'C-N-CA-HA',
            'N-CA-CB-HB1',
            'N-CA-CB-HB2',
            'N-CA-CB-HB3'],
        'torsion-vals': [
            'p',
            # Hydrogens
            ('hi', 'CB', 2 * PI / 3),  # HA
            0,                         # HB1, chi HBX are arbitrary - spaced out by 2pi/3
            2 * PI/3,                  # HB2
            -2 * PI/3,                 # HB3
        ]
    },
    'ARG': {
        'torsion-names': [
            'C-N-CA-CB',
            'N-CA-CB-CG',
            'CA-CB-CG-CD',
            'CB-CG-CD-NE',
            'CG-CD-NE-CZ',
            'CD-NE-CZ-NH1',
            'CD-NE-CZ-NH2',
            # Hydrogens
            'C-N-CA-HA',  # HA
            'N-CA-CB-HB2',
            'N-CA-CB-HB3',
            'CA-CB-CG-HG2',
            'CA-CB-CG-HG3',
            'CB-CG-CD-HD2',
            'CB-CG-CD-HD3',
            'CG-CD-NE-HE',
            'NE-CZ-NH1-HH11',
            'NE-CZ-NH1-HH12',
            'NE-CZ-NH2-HH21',
            'NE-CZ-NH2-HH22',
        ],

        'torsion-vals': [
            'p',
            'p',
            'p',
            'p',
            'p',
            'p',
            'i',
            # Hydrogens
            ('hi', 'CB', 2 * PI / 3),    # HA
            ('hi', 'CG', 2 * PI / 3),    # HB2
            ('hi', 'CG', -2 * PI / 3),   # HB3
            ('hi', 'CD', 2 * PI / 3),    # HG2
            ('hi', 'CD', -2 * PI / 3),   # HG3
            ('hi', 'NE', 2 * PI / 3),    # HD2
            ('hi', 'NE', -2 * PI / 3),   # HD3
            ('hi', 'CZ', -PI),           # HE
            0,                           # HH11
            PI,                          # HH12
            0,                           # HH21
            PI,                          # HH22
            ]
    },
    'ASN': {
        'torsion-names': [
            'C-N-CA-CB',
            'N-CA-CB-CG',
            'CA-CB-CG-OD1',
            'CA-CB-CG-ND2',
            # Hydrogens
            'C-N-CA-HA',  # HA
            'N-CA-CB-HB2',
            'N-CA-CB-HB3',
            'CB-CG-ND2-HD21',
            'CB-CG-ND2-HD22',
            ],
        'torsion-vals': [
            'p',                        # CB
            'p',                        # CG
            'p',                        # OD1
            'i',                        # ND2
            ('hi', 'CB', 2 * PI / 3),   # HA
            ('hi', 'CG', 2 * PI / 3),   # HB2
            ('hi', 'CG', -2 * PI / 3),  # HB3
            0,                          # HD21
            PI,                         # HD22
            ]
    },
    'ASP': {
        'torsion-names': [
            'C-N-CA-CB',
            'N-CA-CB-CG',
            'CA-CB-CG-OD1',
            'CA-CB-CG-OD2',
            # Hydrogens
            'C-N-CA-HA',  # HA
            'N-CA-CB-HB2',
            'N-CA-CB-HB3'
        ],
        'torsion-vals': [
            'p',
            'p',
            'p',
            'i',
            # Hydrogens
            ('hi', 'CB', 2 * PI / 3),   # HA
            ('hi', 'CG', +2 * PI / 3),  # HB2
            ('hi', 'CG', -2 * PI / 3)   # HB3
        ]
    },
    'CYS': {
        'torsion-names': [
            'C-N-CA-CB',
            'N-CA-CB-SG',
            # Hydrogens
            'C-N-CA-HA',  # HA
            'N-CA-CB-HB2',
            'N-CA-CB-HB3',
            'CA-CB-SG-HG'
        ],
        'torsion-vals': [
            'p',                        # CB
            'p',                        # SG
            # Hydrogens
            ('hi', 'CB', 2 * PI / 3),   # HA
            ('hi', 'SG', 2 * PI / 3),   # HB2
            ('hi', 'SG', -2 * PI / 3),  # HB3
            PI,                         # HG
        ]
    },
    'GLN': {
        'torsion-names': [
            'C-N-CA-CB',
            'N-CA-CB-CG',
            'CA-CB-CG-CD',
            'CB-CG-CD-OE1',
            'CB-CG-CD-NE2',
            # Hydrogens
            'C-N-CA-HA',  # HA
            'N-CA-CB-HB2',
            'N-CA-CB-HB3',
            'CA-CB-CG-HG2',
            'CA-CB-CG-HG3',
            'CG-CD-NE2-HE21',
            'CG-CD-NE2-HE22',
        ],
        'torsion-vals': [
            'p',                        # CB
            'p',                        # CG
            'p',                        # CD
            'p',                        # OE1
            'i',                        # NE2
            # Hydrogens
            ('hi', 'CB', 2 * PI / 3),   # HA
            ('hi', 'CG', 2 * PI / 3),   # HB2
            ('hi', 'CG', -2 * PI / 3),  # HB3
            ('hi', 'CD', 2 * PI / 3),   # HG2
            ('hi', 'CD', -2 * PI / 3),  # HG3
            0,                          # HE21
            PI,                         # HE22
        ]
    },
    'GLU': {
        'torsion-names': [
            'C-N-CA-CB',
            'N-CA-CB-CG',
            'CA-CB-CG-CD',
            'CB-CG-CD-OE1',
            'CB-CG-CD-OE2',
            # Hydrogens
            'C-N-CA-HA',  # HA
            'N-CA-CB-HB2',
            'N-CA-CB-HB3',
            'CA-CB-CG-HG2',
            'CA-CB-CG-HG3',
        ],
        'torsion-vals': [
            'p',                        # CB
            'p',                        # CG
            'p',                        # CD
            'p',                        # OE1
            'i',                        # OE2
            # Hydrogens
            ('hi', 'CB', 2 * PI / 3),   # HA
            ('hi', 'CG', 2 * PI / 3),   # HB2
            ('hi', 'CG', -2 * PI / 3),  # HB3
            ('hi', 'CD', 2 * PI / 3),   # HG2
            ('hi', 'CD', -2 * PI / 3)   # HG3
        ]
    },
    'GLY': {
        'torsion-names': [
            'C-N-CA-HA2',  # HA2
            'C-N-CA-HA3',  # HA3
        ],
        'torsion-vals': [
            ('hi', 'phi', 2 * PI / 3),    # HA2
            ('hi', 'phi', -2 * PI / 3),   # HA3

        ]
    },
    'HIS': {                            # Actually HID
        'torsion-names': [
            'C-N-CA-CB',                # CB
            'N-CA-CB-CG',               # CG
            'CA-CB-CG-ND1',             # ND1
            'CB-CG-ND1-CE1',            # CE1
            'CG-ND1-CE1-NE2',           # NE2
            'ND1-CE1-NE2-CD2',          # CD2
            # Hydrogens
            'C-N-CA-HA',                # HA
            'N-CA-CB-HB2',              # HB2
            'N-CA-CB-HB3',              # HB3
            'CB-CG-ND1-HD1',            # HD1
            'CG-ND1-CE1-HE1',           # HE1
            'CE1-NE2-CD2-HD2',          # HD2

        ],
        'torsion-vals': [
            'p',                         # CB
            'p',                         # CG
            'p',                         # ND1
            PI,                          # CE1
            0.0,                         # NE2
            0.0,                         # CD2
            # Hydrogens
            ('hi', 'CB', 2 * PI / 3),    # HA
            ('hi', 'CG', 2 * PI / 3),    # HB2
            ('hi', 'CG', -2 * PI / 3),   # HB3
            0,                           # HD1
            PI,                          # HE1
            PI,                          # HD2
        ]
    },
    'ILE': {
        'torsion-names': [
            'C-N-CA-CB',
            'N-CA-CB-CG1',
            'CA-CB-CG1-CD1',
            'N-CA-CB-CG2',
            'C-N-CA-HA',                 # HA
            'N-CA-CB-HB',                # HB
            'CA-CB-CG1-HG12',            # HG12
            'CA-CB-CG1-HG13',            # HG13
            'CB-CG1-CD1-HD11',           # HD11
            'CB-CG1-CD1-HD12',           # HD12
            'CB-CG1-CD1-HD13',           # HD13
            'CA-CB-CG2-HG21',            # HG21
            'CA-CB-CG2-HG22',            # HG22
            'CA-CB-CG2-HG23',            # HG23
        ],
        'torsion-vals': [
            'p',
            'p',
            'p',
            'p',
            ('hi', 'CB', 2 * PI / 3),    # HA
            ('hi', 'CG1', -2 * PI / 3),  # HB
            ('hi', 'CD1', 2 * PI / 3),   # HG12
            ('hi', 'CD1', -2 * PI / 3),  # HG13
            0,                           # HD11
            2*PI/3,                      # HD12
            -2*PI/3,                     # HD13
            0,                           # HG21
            2*PI/3,                      # HG22
            -2*PI/3,                     # HG23
        ]
    },
    'LEU': {
        'torsion-names': [
            'C-N-CA-CB',                # CB
            'N-CA-CB-CG',               # CG
            'CA-CB-CG-CD1',             # CD1
            'CA-CB-CG-CD2',             # CD2
            'C-N-CA-HA',                # HA
            'N-CA-CB-HB2',              # HB2
            'N-CA-CB-HB3',              # HB3
            'CA-CB-CG-HG',              # HG
            'CB-CG-CD1-HD11',           # HD11
            'CB-CG-CD1-HD12',           # HD12
            'CB-CG-CD1-HD13',           # HD13
            'CB-CG-CD2-HD21',           # HD21
            'CB-CG-CD2-HD22',           # HD22
            'CB-CG-CD2-HD23',           # HD23
        ],
        'torsion-vals': [
            # chi HB2 is defined by chi used to place CG rotated +/- 2pi/3  # HB2
            # chi HD1X and HD2X are arbitrary - spaced out by 2pi/3
            'p',                         # CB
            'p',                         # CG
            'p',                         # CD1
            ('hi', 'CD1', -2 * PI / 3),  # CD2,  # TODO Do not measure/predict this angle
            ('hi', 'CB', 2 * PI / 3),    # HA
            ('hi', 'CG', 2 * PI / 3),    # HB2
            ('hi', 'CG', -2 * PI / 3),   # HB3
            ('hi', 'CD1', 2 * PI / 3),   # HG
            0,                           # HD11
            2.09439510239,               # HD12
            -2.09439510239,              # HD13
            0,                           # HD21
            2.09439510239,               # HD22
            -2.09439510239,              # HD23
        ]
    },
    'LYS': {
        'torsion-names': [
            'C-N-CA-CB',
            'N-CA-CB-CG',
            'CA-CB-CG-CD',
            'CB-CG-CD-CE',
            'CG-CD-CE-NZ',
            'C-N-CA-HA',                # HA
            'N-CA-CB-HB2',              # HB2
            'N-CA-CB-HB3',              # HB3
            'CA-CB-CG-HG2',             # HG2
            'CA-CB-CG-HG3',             # HG3
            'CB-CG-CD-HD2',             # HD2
            'CB-CG-CD-HD3',             # HD3
            'CG-CD-CE-HE2',             # HE2
            'CG-CD-CE-HE3',             # HE3
            'CD-CE-NZ-HZ1',             # HZ1
            'CD-CE-NZ-HZ2',             # HZ2
            'CD-CE-NZ-HZ3',             # HZ3
        ],
        'torsion-vals': [
            'p',
            'p',
            'p',
            'p',
            'p',
            ('hi', 'CB', 2 * PI / 3),   # HA
            ('hi', 'CG', 2 * PI / 3),   # HB2
            ('hi', 'CG', -2 * PI / 3),  # HB3
            ('hi', 'CD', 2 * PI / 3),   # HG2
            ('hi', 'CD', -2 * PI / 3),  # HG3
            ('hi', 'CE', 2 * PI / 3),   # HD2
            ('hi', 'CE', -2 * PI / 3),  # HD3
            ('hi', 'NZ', 2 * PI / 3),   # HE2
            ('hi', 'NZ', -2 * PI / 3),  # HE3
            0,                          # HZ1
            -2*PI/3,                    # HZ2
            2*PI/3,                     # HZ3
            ]
    },
    'MET': {
        'torsion-names': [
            'C-N-CA-CB',
            'N-CA-CB-CG',
            'CA-CB-CG-SD',
            'CB-CG-SD-CE',
            'C-N-CA-HA',                # HA
            'N-CA-CB-HB2',              # HB2
            'N-CA-CB-HB3',              # HB3
            'CA-CB-CG-HG2',             # HG2
            'CA-CB-CG-HG3',             # HG3
            'CG-SD-CE-HE1',  # HE1
            'CG-SD-CE-HE2',  # HE2
            'CG-SD-CE-HE3',  # HE3
            ],
        'torsion-vals': [
            'p',
            'p',
            'p',
            'p',
            ('hi', 'CB', 2 * PI / 3),   # HA
            ('hi', 'CG', 2 * PI / 3),   # HB2
            ('hi', 'CG', -2 * PI / 3),  # HB3
            ('hi', 'SD', 2 * PI / 3),   # HG2
            ('hi', 'SD', -2 * PI / 3),  # HG3
            PI/3 + 0,                   # HE1
            PI/3 + 2*PI/3,              # HE2
            PI/3 - 2*PI/3,              # HE3
            ]
    },
    'PHE': {
        'torsion-names': [
            'C-N-CA-CB',
            'N-CA-CB-CG',
            'CA-CB-CG-CD1',
            'CB-CG-CD1-CE1',
            'CG-CD1-CE1-CZ',
            'CD1-CE1-CZ-CE2',
            'CE1-CZ-CE2-CD2',
            'C-N-CA-HA',                # HA
            'N-CA-CB-HB2',              # HB2
            'N-CA-CB-HB3',              # HB3
            'CB-CG-CD1-HD1',            # HD1
            'CG-CD1-CE1-HE1',           # HE1
            'CD1-CE1-CZ-HZ',            # HZ
            'CE1-CZ-CE2-HE2',           # HE2
            'CZ-CE2-CD2-HD2',           # HD2
        ],
        'torsion-vals': [
            'p',
            'p',
            'p',
            PI,
            0.0,
            0.0,
            0.0,
            ('hi', 'CB', 2 * PI / 3),   # HA
            ('hi', 'CG', 2 * PI / 3),   # HB2
            ('hi', 'CG', -2 * PI / 3),  # HB3
            0,                          # HD1
            PI,                         # HE1
            PI,                         # HZ
            PI,                         # HE2
            PI,                         # HD2
        ]
    },
    'PRO': {
        'torsion-names': [
            'C-N-CA-CB',
            'N-CA-CB-CG',
            'CA-CB-CG-CD',
            'C-N-CA-HA',                # HA
            'N-CA-CB-HB2',              # HB2
            'N-CA-CB-HB3',              # HB3
            'CA-CB-CG-HG2',             # HG2
            'CA-CB-CG-HG3',             # HG3
            'CB-CG-CD-HD2',             # HD2
            'CB-CG-CD-HD3',             # HD3
        ],
        'torsion-vals': [
            'p',
            'p',
            'p',
            ('hi', 'CB', 2 * PI / 3),   # HA
            ('hi', 'CG', 2 * PI / 3),   # HB2
            ('hi', 'CG', -2 * PI / 3),  # HB3
            ('hi', 'CD', 2 * PI / 3),   # HG2
            ('hi', 'CD', -2 * PI / 3),  # HG3
            2 * PI / 3,                 # HD2
            -2 * PI / 3,                # HD3
        ]
    },
    'SER': {
        'torsion-names': [
            'C-N-CA-CB',
            'N-CA-CB-OG',
            'C-N-CA-HA',                # HA
            'N-CA-CB-HB2',              # HB2
            'N-CA-CB-HB3',              # HB3
            'CA-CB-OG-HG',              # HG
        ],
        'torsion-vals': [
            'p',
            'p',
            ('hi', 'CB', 2 * PI / 3),   # HA
            ('hi', 'OG', 2 * PI / 3),   # HB2
            ('hi', 'OG', -2 * PI / 3),  # HB3
            PI,                         # HG
        ]
    },
    'THR': {
        'torsion-names': [
            'C-N-CA-CB',
            'N-CA-CB-OG1',
            'N-CA-CB-CG2',
            'C-N-CA-HA',                 # HA
            'N-CA-CB-HB',                # HB
            'CA-CB-OG1-HG1',             # HG1
            'CA-CB-CG2-HG21',            # HG21
            'CA-CB-CG2-HG22',            # HG22
            'CA-CB-CG2-HG23',            # HG23
        ],
        'torsion-vals': [
            'p',
            'p',
            'p',
            ('hi', 'CB', 2 * PI / 3),    # HA
            ('hi', 'OG1', -2 * PI / 3),  # HB
            PI,                          # HG1  
            0,                           # HG21
            2*PI/3,                      # HG22
            -2*PI/3,                     # HG23
        ]
    },
    'TRP': {
        'torsion-names': [  # TODO cw-na-h is not defined for trp, so we use CN-NA-H
            'C-N-CA-CB',
            'N-CA-CB-CG',
            'CA-CB-CG-CD1',
            'CB-CG-CD1-NE1',
            'CG-CD1-NE1-CE2',
            'CD1-NE1-CE2-CZ2',
            'NE1-CE2-CZ2-CH2',
            'CE2-CZ2-CH2-CZ3',
            'CZ2-CH2-CZ3-CE3',
            'CH2-CZ3-CE3-CD2',
            'C-N-CA-HA',  # HA
            'N-CA-CB-HB2',
            'N-CA-CB-HB3',
            'CB-CG-CD1-HD1',
            'CZ2-CE2-NE1-HE1',
            'NE1-CE2-CZ2-HZ2',
            'CE2-CZ2-CH2-HH2',
            'CZ2-CH2-CZ3-HZ3',
            'CH2-CZ3-CE3-HE3',
        ],
        'torsion-vals': [
            'p',                        # CB
            'p',                        # CG
            'p',                        # CD1
            PI,                         # NE1
            0.0,                        # CE2
            PI,                         # CZ2
            PI,                         # CH2
            0.0,                        # CZ3
            0.0,                        # CE3
            0.0,                        # CD2
            ('hi', 'CB', 2 * PI / 3),   # HA
            ('hi', 'CG', 2 * PI / 3),   # HB2
            ('hi', 'CG', -2 * PI / 3),  # HB3
            0,                          # HD1
            PI,                         # HE1
            0,                          # HZ2
            PI,                         # HH2
            PI,                         # HZ3
            PI,                         # HE3
            ]
    },
    'TYR': {
        'torsion-names': [
            'C-N-CA-CB',
            'N-CA-CB-CG',
            'CA-CB-CG-CD1',
            'CB-CG-CD1-CE1',
            'CG-CD1-CE1-CZ',
            'CD1-CE1-CZ-OH',
            'CD1-CE1-CZ-CE2',
            'CE1-CZ-CE2-CD2',
            'C-N-CA-HA',                # HA
            'N-CA-CB-HB2',              # HB2
            'N-CA-CB-HB3',              # HB3
            'CB-CG-CD1-HD1',            # HD1
            'CG-CD1-CE1-HE1',           # HE1
            'CE1-CZ-OH-HH',             # HH
            'CE1-CZ-CE2-HE2',           # HE2
            'CZ-CE2-CD2-HD2',           # HD2
        ],
        'torsion-vals': [
            'p',
            'p',
            'p',
            PI,
            0.0,
            PI,
            0.0,
            0.0,
            ('hi', 'CB', 2 * PI / 3),   # HA
            ('hi', 'CG', 2 * PI / 3),   # HB2
            ('hi', 'CG', -2 * PI / 3),  # HB3
            0,                          # HD1
            PI,                         # HE1
            PI,                         # HH
            PI,                         # HE2
            PI,                         # HD2
        ]
    },
    'VAL': {
        'torsion-names': [
            'C-N-CA-CB',
            'N-CA-CB-CG1',
            'N-CA-CB-CG2',
            'C-N-CA-HA',                 # HA
            'N-CA-CB-HB',                # HB
            'CA-CB-CG1-HG11',            # HG11
            'CA-CB-CG1-HG12',            # HG12
            'CA-CB-CG1-HG13',            # HG13
            'CA-CB-CG2-HG21',            # HG21
            'CA-CB-CG2-HG22',            # HG22
            'CA-CB-CG2-HG23',            # HG23
            ],
        'torsion-vals': [
            'p',
            'p',
            'p',
            ('hi', 'CB', 2 * PI / 3),    # HA
            ('hi', 'CG1', 2 * PI / 3),   # HB
            0,                           # HG11
            2*PI/3,                      # HG12
            -2*PI/3,                     # HG13
            0,                           # HG21
            2*PI/3,                      # HG22
            -2*PI/3,                     # HG23
            ]
    }
}
# yapf: enable

# Add atom names to build dict
for resname, res_info in _RAW_BUILD_INFO.items():
    atom_names = [torsion.split("-")[-1] for torsion in res_info['torsion-names']]
    _RAW_BUILD_INFO[resname]['atom-names'] = atom_names

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


def _get_max_num_atoms_heavy():
    lengths = []
    for resname, resinfo in _RAW_BUILD_INFO.items():
        heavyatms = list(filter(lambda an: not an.startswith("H"), resinfo['atom-names']))
        nonh_terminals = list(
            filter(lambda an: not an.startswith('H'), _SUPPORTED_TERMINAL_ATOMS))
        n = len(heavyatms) + len(nonh_terminals) + len(_BACKBONE_ATOMS)
        lengths.append(n)
    return max(lengths)


def _get_max_num_atoms_hydrogen():
    lengths = []
    for resname, resinfo in _RAW_BUILD_INFO.items():
        n = len(resinfo['torsion-names']) + len(_SUPPORTED_TERMINAL_ATOMS) + len(
            _BACKBONE_ATOMS + ['H'])
        lengths.append(n)
    return max(lengths)


_SUPPORTED_TERMINAL_ATOMS = ['OXT', 'H2', 'H3']
_BACKBONE_ATOMS = ['N', 'CA', 'C', 'O']
_MAX_NUM_ATOMS_HEAVY = _get_max_num_atoms_heavy()
_MAX_NUM_ATOMS_HYDROGEN = _get_max_num_atoms_hydrogen()
assert _MAX_NUM_ATOMS_HYDROGEN == 27, _MAX_NUM_ATOMS_HYDROGEN
assert _MAX_NUM_ATOMS_HEAVY == 15, _MAX_NUM_ATOMS_HEAVY
NUM_COORDS_PER_RES = _MAX_NUM_ATOMS_HEAVY
NUM_COORDS_PER_RES_W_HYDROGENS = _MAX_NUM_ATOMS_HYDROGEN


def _create_atomname_lookup():
    """Create a dictionary mapping resname3 to dict mapping atom_name to atom_type."""
    amino12 = pkg_resources.resource_filename("sidechainnet",
                                              "resources/amino12.lib")
    with open(amino12, "r") as f:
        text = f.read()
    atom_name_lookup = {}
    amino_name_info = re.findall(
        r"!entry\.[A-Z]{3}\.unit\.atoms table[\s\S]+?(?=!entry\.[A-Z]{3})", text, re.S)
    for match in amino_name_info:
        lines = match.split("\n")
        resname = lines[0].split(".")[1]
        assert len(resname) != 4, "terminal res not supported"
        if resname == 'HID':
            resname = 'HIS'
        if resname not in _RAW_BUILD_INFO:
            continue
        atom_name_lookup[resname] = {}
        for line in lines[1:]:
            if not line:
                continue
            atom_data = line.split()
            a1, a2 = atom_data[:2]
            a1 = a1.replace("\"", "")
            a2 = a2.replace("\"", "")
            atom_name_lookup[resname][a1] = a2
    return atom_name_lookup


class ForceFieldLookupHelper(object):
    """Reads in AMBER ForceField data to facilitate bond/angle lookups with Regex."""

    def __init__(self, *forcefield_files):
        """Ingest one or more forcefield files ordered in terms of priority."""
        self.ff_files = forcefield_files
        self.text = ""

        for fn in self.ff_files:
            with open(fn, "r") as f:
                self.text += f.read()

        # Here we make some pre-emptive corrections for values that are not adequate in
        # AMBER. AMBER seems to specify interior angles for histidine and proline wich are
        # much too large (120 degrees and 109.5 degrees, respectively). Here, I have
        # remeasured the offending values.

        self.overwrite_rules = {}
        histidine_interior_angles = ["CT-CC-NA", "CC-NA-CR", "NA-CR-NB", "CR-NB-CV"]
        self.overwrite_rules[('HIS', histidine_interior_angles[0])] = np.deg2rad(122.6690)
        for hia in histidine_interior_angles[1:]:
            self.overwrite_rules[('HIS', hia)] = np.deg2rad(108)
        self.overwrite_rules[('PRO', "N -CX-CT")] = np.deg2rad(101.8812)
        self.overwrite_rules[('PRO', "CX-CT-CT")] = np.deg2rad(103.6465)
        self.overwrite_rules[('PRO', "CX-CT-CT")] = np.deg2rad(103.3208)

    def get_value(self, resname, item, convert_to_rad=True):
        """Lookup the value (bond length or bond angle) for the item (ex 'CX-2C':1.526).

        Args:
            item (str): A string representing a bond (2 atoms) or angle (3 atoms) with
                atom types separated by hyphens.

        Returns:
            value (float): The value of item as specified in AMBER forcefield parameters.
        """
        assert len(item.split("-")) <= 3, "Does not support torsion lookup."

        if (resname, item) in self.overwrite_rules:
            return self.overwrite_rules[(resname, item)]

        def get_match(s):
            s = s.replace("*", "\\*")
            pattern = f"^{s}[^-].+"
            return re.search(pattern, self.text, re.MULTILINE)

        match = get_match(item)

        if match is None:
            # Second attempt - look for the reverse item
            item = "-".join(item.split("-")[::-1])
            match = get_match(item)
            if match is None:
                raise ValueError(f"No match found for {item}.")

        line = match.group(0)
        line = line.replace(item, "").strip()
        value = float(line.split()[1])
        if convert_to_rad and len(item.split("-")) == 3:
            return np.deg2rad(value)
        return value


def create_complete_hydrogen_build_info_dict():
    """Create a Python dictionary that describes all necessary info for building atoms."""
    # We need to record the following information: bond lengths, bond angles
    new_build_info = copy.deepcopy(_RAW_BUILD_INFO)

    # First, we must lookup the atom type for each atom
    atom_type_map = _create_atomname_lookup()

    # We also make ready a way to lookup forcefield parameter values from raw files
    frcmod = pkg_resources.resource_filename("sidechainnet", "resources/frcmod.ff14SB")
    parm10 = pkg_resources.resource_filename("sidechainnet", "resources/parm10.dat")
    ff_helper = ForceFieldLookupHelper(frcmod, parm10)

    # Now, let's generate the types of the bonds and angles we will need to lookup
    for resname, resinfo in _RAW_BUILD_INFO.items():
        new_build_info[resname]['bond-types'] = []
        new_build_info[resname]['angle-types'] = []
        new_build_info[resname]['bond-vals'] = []
        new_build_info[resname]['angle-vals'] = []
        for torsion in resinfo['torsion-names']:
            atomtypes = [f"{atom_type_map[resname][an]:<2}" for an in torsion.split("-")]
            bond = "-".join(atomtypes[-2:])
            angle = "-".join(atomtypes[-3:])
            new_build_info[resname]['bond-types'].append(bond)
            new_build_info[resname]['angle-types'].append(angle)

            # Given names/types of necessary bonds/angles, we look up their AMBER vals
            new_build_info[resname]['bond-vals'].append(ff_helper.get_value(
                resname, bond))
            new_build_info[resname]['angle-vals'].append(
                ff_helper.get_value(resname, angle))

    return new_build_info


SC_HBUILD_INFO = create_complete_hydrogen_build_info_dict()
from sidechainnet.structure.fastbuild import AA1to3

ATOM_MAP_HEAVY = {}
for a, aaa in AA1to3.items():
    if len(aaa) == 4:  # terminal res
        continue
    atom_names = list(
        filter(lambda x: not x.startswith("H"), SC_HBUILD_INFO[aaa]['atom-names']))
    ATOM_MAP_HEAVY[a] = ["N", "CA", "C", "O", "OXT"] + atom_names
    ATOM_MAP_HEAVY[a].extend(["PAD"] * (NUM_COORDS_PER_RES - len(ATOM_MAP_HEAVY[a])))

ATOM_MAP_H = {}
for a, aaa in AA1to3.items():
    if len(aaa) == 4:  # terminal res
        continue
    ATOM_MAP_H[a] = ["N", "CA", "C", "O", "OXT", "H", "H2", "H3"] + SC_HBUILD_INFO[aaa]['atom-names']
    ATOM_MAP_H[a].extend(["PAD"] * (NUM_COORDS_PER_RES_W_HYDROGENS - len(ATOM_MAP_H[a])))

if __name__ == "__main__":
    import pprint
    # import json
    hbp = create_complete_hydrogen_build_info_dict()
    hbp_str = pprint.pformat(hbp, indent=1, width=90, compact=False)
    with open("hbp.py", "w") as f:
        # f.write(json.dumps(hbp, indent=2))
        f.write(hbp_str)
