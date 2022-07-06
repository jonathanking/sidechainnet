import numpy as np

from sidechainnet.dataloaders.SCNProtein import SCNProtein
from sidechainnet.structure.build_info import NUM_ANGLES, NUM_COORDS_PER_RES
from sidechainnet.utils.sequence import ONE_TO_THREE_LETTER_MAP

# TODO update test so that coordinate dims are L x ncoords x 3.
def test_trim_edges():
    seq = "THISISATESTPRQTEIN"  # L = 18
    msk = "-------++++-------"
    crd = np.ones((18*NUM_COORDS_PER_RES, 3))
    ang = np.ones((18, NUM_ANGLES))
    ums = " ".join([ONE_TO_THREE_LETTER_MAP[c] for c in seq])
    p = SCNProtein(sequence=seq, mask=msk, coordinates=crd, angles=ang, unmodified_seq=ums, id="TEST")

    p.trim_edges()

    assert len(p.seq) == 4
    assert p.seq == "TEST"
    assert p.coords.shape[0] == 4*NUM_COORDS_PER_RES

    p.add_hydrogens()

    assert p.hcoords.shape[0] == 4*p.atoms_per_res
    assert len(p.unmodified_seq.split()) == 4
    assert len(p.mask) == 4