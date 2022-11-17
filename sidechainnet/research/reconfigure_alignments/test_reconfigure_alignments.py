
from types import SimpleNamespace
from sidechainnet.research.reconfigure_alignments.reconfigure_alignments import load_alignment_via_openfold, load_alignment_with_bio


def test_load_alignment1():
    fn = "/scr/roda/pdb/2qdv_A/a3m/bfd_uniclust_hits.a3m"
    # alignment = load_alignment(fn)
    a2 = load_alignment_with_bio(fn)

def test_align_and_select():
    fn = "/scr/roda/pdb/2qdv_A/a3m/bfd_uniclust_hits.a3m"
    alignment = load_alignment_with_bio(fn)
    p = SimpleNamespace
    print(alignment)