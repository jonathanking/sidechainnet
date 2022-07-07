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
import numpy as np

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

# TODO Consolidate info: organize by atom name, remove types fields for all but torsions

# yapf: disable
SC_HBUILD_INFO = {
    'ALA': {
        'angles-vals': [1.9146261894377796],
        'bonds-vals': [1.526],
        'torsion-names': ['C-N-CA-CB'],
        'torsion-types': ['C -N -CX-CT'],
        'torsion-vals': ['p']
    },
    'ARG': {

        'angles-vals': [
            1.9146261894377796, 1.911135530933791, 1.911135530933791, 1.9408061282176945,
            2.150245638457014, 2.0943951023931953, 2.0943951023931953
        ],
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
        'angles-vals': [
            1.9146261894377796, 1.9390607989657, 2.101376419401173, 2.035053907825388
        ],
        'bonds-vals': [1.526, 1.522, 1.229, 1.335],
        'torsion-names': ['C-N-CA-CB', 'N-CA-CB-CG', 'CA-CB-CG-OD1', 'CA-CB-CG-ND2'],
        'torsion-types': ['C -N -CX-2C', 'N -CX-2C-C ', 'CX-2C-C -O ', 'CX-2C-C -N '],
        'torsion-vals': ['p', 'p', 'p', 'i']
    },
    'ASP': {
        'angles-vals': [
            1.9146261894377796,
            1.9390607989657,
            2.0420352248333655,
            2.0420352248333655,

            1.9111355,
            1.9111355
        ],

        'bonds-vals': [
            1.526,
            1.522,
            1.25,
            1.25,

            1.09,
            1.09],
        'torsion-names': [
            'C-N-CA-CB',
            'N-CA-CB-CG',
            'CA-CB-CG-OD1',
            'CA-CB-CG-OD2',

            'N-CA-CB-HB2',
            'N-CA-CB-HB3'],
        'torsion-types': [
            'C -N -CX-2C',
            'N -CX-2C-CO',
            'CX-2C-CO-O2',
            'CX-2C-CO-O2',

            'N -CX-2C-HC',
            'N -CX-2C-HC'],
        'torsion-vals': [
            'p',
            'p',
            'p',
            'i',

            ('hi', 'CG', +2*np.pi/3),
            ('hi', 'CG', -2*np.pi/3)]
    },
    'CYS': {
        'angles-vals': [1.9146261894377796, 1.8954275676658419],
        'bonds-vals': [1.526, 1.81],
        'torsion-names': ['C-N-CA-CB', 'N-CA-CB-SG'],
        'torsion-types': ['C -N -CX-2C', 'N -CX-2C-SH'],
        'torsion-vals': ['p', 'p']
    },
    'GLN': {
        'angles-vals': [
            1.9146261894377796, 1.911135530933791, 1.9390607989657, 2.101376419401173,
            2.035053907825388
        ],
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
        'angles-vals': [
            1.9146261894377796,
            1.911135530933791,
            1.9390607989657,
            2.0420352248333655,
            2.0420352248333655,

            1.9111355,
            1.9111355,
            1.9111355,
            1.9111355,
        ],
        'bonds-vals': [
            1.526,
            1.526,
            1.522,
            1.25,
            1.25,

            1.0900,
            1.0900,
            1.0900,
            1.0900],
        'torsion-names': [
            'C-N-CA-CB',
            'N-CA-CB-CG',
            'CA-CB-CG-CD',
            'CB-CG-CD-OE1',
            'CB-CG-CD-OE2',

            'N-CA-CB-HB2',
            'N-CA-CB-HB3',
            'CA-CB-CG-HG2',
            'CA-CB-CG-HG3'
        ],
        'torsion-types': [
            'C -N -CX-2C',
            'N -CX-2C-2C',
            'CX-2C-2C-CO',
            '2C-2C-CO-O2',
            '2C-2C-CO-O2',

            'N -CX-2C-HC',
            'N -CX-2C-HC',
            'CX-2C-2C-HC',
            'CX-2C-2C-HC'
        ],
        'torsion-vals': [
            'p',
            'p',
            'p',
            'p',
            'i',

            ('hi', 'CG',  2*np.pi/3),  # chi HB2 is defined by chi used to place CG rotated +/- 2pi/3
            ('hi', 'CG', -2*np.pi/3),
            ('hi', 'CD',  2*np.pi/3),
            ('hi', 'CD', -2*np.pi/3)]
    },
    'GLY': {
        'angles-vals': [],
        'bonds-vals': [],
        'torsion-names': [],
        'torsion-types': [],
        'torsion-vals': []
    },
    'HIS': {
        'angles-vals': [
            1.9146261894377796, 1.9739673840055867, 2.0943951023931953,
            1.8849555921538759, 1.8849555921538759, 1.8849555921538759
        ],
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
        'angles-vals': [
            1.9146261894377796, 1.911135530933791, 1.911135530933791, 1.911135530933791
        ],
        'bonds-vals': [1.526, 1.526, 1.526, 1.526],
        'torsion-names': ['C-N-CA-CB', 'N-CA-CB-CG1', 'CA-CB-CG1-CD1', 'N-CA-CB-CG2'],
        'torsion-types': ['C -N -CX-3C', 'N -CX-3C-2C', 'CX-3C-2C-CT', 'N -CX-3C-CT'],
        'torsion-vals': ['p', 'p', 'p', 'p']
    },
    'LEU': {

        'angles-vals': [
            1.9146261894377796,
            1.911135530933791,
            1.911135530933791,
            1.911135530933791,

            1.9111355,  # HA

            1.9111355,
            1.9111355,
            1.9111355,
            1.9111355,
            1.9111355,
            1.9111355,
            1.9111355,
            1.9111355,
            1.9111355,
        ],

        'bonds-vals': [
            1.526,
            1.526,
            1.526,
            1.526,

            1.090,  # HA

            1.0900,
            1.0900,
            1.090,
            1.090,
            1.090,
            1.090,
            1.090,
            1.090,
            1.090],
        'torsion-names': [
            'C-N-CA-CB',
            'N-CA-CB-CG',
            'CA-CB-CG-CD1',
            'CA-CB-CG-CD2',

            'C-N-CA-HA',  # HA

            'N-CA-CB-HB2',
            'N-CA-CB-HB3',
            'CB-CG-CD1-HD11',
            'CB-CG-CD1-HD12',
            'CB-CG-CD1-HD13',
            'CB-CG-CD2-HD21',
            'CB-CG-CD2-HD22',
            'CB-CG-CD2-HD23',
            'CA-CB-CG-HG'],
        'torsion-types': [
            'C -N -CX-2C',
            'N -CX-2C-3C',
            'CX-2C-3C-CT',
            'CX-2C-3C-CT',

            'C -N -CX-H1',  # HA

            'N -CX-2C-HC',
            'N -CX-2C-HC',
            '2C-3C-CT-HC',
            '2C-3C-CT-HC',
            '2C-3C-CT-HC',
            '2C-3C-CT-HC',
            '2C-3C-CT-HC',
            '2C-3C-CT-HC',
            'CX-2C-3C-HC'],
        'torsion-vals': [
            'p',
            'p',
            'p',
            'p',

            ('hi', 'CB', 2*np.pi/3),  # HA

            ('hi', 'CG',  2*np.pi/3),  # chi HB2 is defined by chi used to place CG rotated +/- 2pi/3
            ('hi', 'CG', -2*np.pi/3),
            0,               # chi HD1X are arbitrary - spaced out by 2pi/3
            2.09439510239,   # 2pi/3
            -2.09439510239,  # -2pi/3
            0,               # chi HD2X are arbitrary - spaced out by 2pi/3
            2.09439510239,
            -2.09439510239,
            ('hi', 'CD1', 2*np.pi/3),
            ]
    },
    'LYS': {
        'angles-vals': [
            1.9146261894377796, 1.911135530933791, 1.911135530933791, 1.911135530933791,
            1.9408061282176945
        ],
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
        'angles-vals': [
            1.9146261894377796, 1.911135530933791, 2.0018926520374962, 1.726130630222392
        ],
        'bonds-vals': [1.526, 1.526, 1.81, 1.81],
        'torsion-names': ['C-N-CA-CB', 'N-CA-CB-CG', 'CA-CB-CG-SD', 'CB-CG-SD-CE'],
        'torsion-types': ['C -N -CX-2C', 'N -CX-2C-2C', 'CX-2C-2C-S ', '2C-2C-S -CT'],
        'torsion-vals': ['p', 'p', 'p', 'p']
    },
    'PHE': {

        'angles-vals': [
            1.9146261894377796, 1.9896753472735358, 2.0943951023931953,
            2.0943951023931953, 2.0943951023931953, 2.0943951023931953, 2.0943951023931953
        ],

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

        'angles-vals': [1.9146261894377796, 1.911135530933791, 1.911135530933791],

        'bonds-vals': [1.526, 1.526, 1.526],
        'torsion-names': ['C-N-CA-CB', 'N-CA-CB-CG', 'CA-CB-CG-CD'],
        'torsion-types': ['C -N -CX-CT', 'N -CX-CT-CT', 'CX-CT-CT-CT'],
        'torsion-vals': ['p', 'p', 'p']
    },
    'SER': {

        'angles-vals': [1.9146261894377796, 1.911135530933791],

        'bonds-vals': [1.526, 1.41],
        'torsion-names': ['C-N-CA-CB', 'N-CA-CB-OG'],
        'torsion-types': ['C -N -CX-2C', 'N -CX-2C-OH'],
        'torsion-vals': ['p', 'p']
    },
    'THR': {

        'angles-vals': [1.9146261894377796, 1.911135530933791, 1.911135530933791],

        'bonds-vals': [1.526, 1.41, 1.526],
        'torsion-names': ['C-N-CA-CB', 'N-CA-CB-OG1', 'N-CA-CB-CG2'],
        'torsion-types': ['C -N -CX-3C', 'N -CX-3C-OH', 'N -CX-3C-CT'],
        'torsion-vals': ['p', 'p', 'p']
    },
    'TRP': {
        'angles-vals': [
            1.9146261894377796,
            2.0176006153054447,
            2.181661564992912,
            1.8971728969178363,
            1.9477874452256716,
            2.3177972466484698,
            2.0943951023931953,
            2.0943951023931953,
            2.0943951023931953,
            2.0943951023931953,

            1.9111355,
            1.9111355,
            -np.deg2rad(120.00),
            -np.deg2rad(123.1),
            np.deg2rad(120.00),
            -np.deg2rad(120.00),
            np.deg2rad(120.00),
            np.deg2rad(120.0)
        ],

        'bonds-vals': [
            1.526,
            1.495,
            1.352,
            1.381,
            1.38,
            1.4,
            1.4,
            1.4,
            1.4,
            1.404,

            1.090,
            1.090,
            1.080,  # TODO this might come from histidine, not sure why
            1.010,  # TODO this might be for gua/ura/his
            1.080,
            1.080,
            1.080,
            1.080],
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

            'N-CA-CB-HB2',
            'N-CA-CB-HB3',
            'CB-CG-CD1-HD1',
            'CZ2-CE2-NE1-HE1',
            'CH2-CZ3-CE3-HE3',
            'CE2-CZ2-CH2-HH2',
            'NE1-CE2-CZ2-HZ2',
            'CZ2-CH2-CZ3-HZ3'
        ],
        'torsion-types': [
            'C -N -CT-CT',
            'N -CT-CT-C*',
            'CT-CT-C*-CW',
            'CT-C*-CW-NA',
            'C*-CW-NA-CN',
            'CW-NA-CN-CA',
            'NA-CN-CA-CA',
            'CN-CA-CA-CA',
            'CA-CA-CA-CA',
            'CA-CA-CA-CB',

            'N -CT-CT-HC',
            'N -CT-CT-HC',
            'CT-C*-CW-H4',
            'CA-CN-NA-H',
            'CA-CA-CA-HA',
            'CN-CA-CA-HA',
            'NA-CN-CA-HA',
            'CA-CA-CA-HA'
        ],
        'torsion-vals': [
            'p',                        #  'CB',
            'p',                        #  'CG',
            'p',                        #  'CD1',
            3.141592653589793,          #  'NE1',
            0.0,                        #  'CE2',
            3.141592653589793,          #  'CZ2',
            3.141592653589793,          #  'CH2',
            0.0,                        #  'CZ3',
            0.0,                        #  'CE3',
            0.0,                        #  'CD2',

            ('hi', 'CG', 2*np.pi/3),    #  'HB2',
            ('hi', 'CG', -2*np.pi/3),   #  'HB3',
            ('hi', 'NE1', -np.pi),       #  'HD1',
            ('hi', 'CE2', -np.pi),      #  'HE1',
            ('hi', 'CD2', -np.pi),      #  'HE3',
            ('hi', 'CZ3', -np.pi),      #  'HH2',
            ('hi', 'CH2', -np.pi),      #  'HZ2',
            ('hi', 'CE3', -np.pi),      #  'HZ3'
        ]
    },
    'TYR': {

        'angles-vals': [
            1.9146261894377796, 1.9896753472735358, 2.0943951023931953,
            2.0943951023931953, 2.0943951023931953, 2.0943951023931953,
            2.0943951023931953, 2.0943951023931953
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

        'angles-vals': [1.9146261894377796, 1.911135530933791, 1.911135530933791],
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
