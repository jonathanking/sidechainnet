from webbrowser import get
import numpy as np

from sidechainnet.dataloaders.SCNProtein import SCNProtein
from sidechainnet.dataloaders.SCNDataset import SCNDataset
from sidechainnet.structure.build_info import NUM_ANGLES, NUM_COORDS_PER_RES
from sidechainnet.utils.sequence import ONE_TO_THREE_LETTER_MAP

from sidechainnet.tests.test_fastbuild import get_p
from sidechainnet.examples.alphabet_protein import get_alphabet_protein


# TODO update test so that coordinate dims are L x ncoords x 3.
def test_trim_edges():
    seq = "THISISATESTPRQTEIN"  # L = 18
    msk = "-------++++-------"
    crd = np.ones((18, NUM_COORDS_PER_RES, 3))
    ang = np.ones((18, NUM_ANGLES))
    ums = " ".join([ONE_TO_THREE_LETTER_MAP[c] for c in seq])
    p = SCNProtein(sequence=seq,
                   mask=msk,
                   coordinates=crd,
                   angles=ang,
                   unmodified_seq=ums,
                   id="TEST")
    p.torch()

    p.trim_edges()

    assert len(p.seq) == 4
    assert p.seq == "TEST"
    assert p.coords.shape[0] == 4

    p.fastbuild(add_hydrogens=True, inplace=True)

    assert p.hcoords.shape[0] == 4
    assert len(p.unmodified_seq.split()) == 4
    assert len(p.mask) == 4


def test_scndataset_delete_ids():
    p1 = get_alphabet_protein()
    p2 = get_alphabet_protein()
    p1.id = 'alphabet1'
    p2.id = 'alphabet2'

    d = SCNDataset.from_scnproteins([p1, p2])
    assert len(d) == 2
    assert 'alphabet1' in d
    d.delete_ids(['alphabet1'])
    assert len(d) == 1
    assert d[0].id == 'alphabet2'
    assert 'alphabet1' not in d



def test_remove_hydrogens():
    p = get_p()
    p.coords = p.coords.reshape(len(p), -1, 3)
    p.hcoords = p.hcoords.reshape(len(p), -1, 3)
    p.fastbuild(add_hydrogens=True, inplace=True)

    c = p.hydrogenrep_to_heavyatomrep(inplace=True)

    assert c.shape[1] == NUM_COORDS_PER_RES


if __name__ == "__main__":
    test_remove_hydrogens()
