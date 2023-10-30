import sidechainnet as scn

def test_load_casp12_30():
    """Tests loading of CASP12 30-thinning proteins."""
    scn.load(casp_version=12, thinning=30)


def test_load_caspdebug():
    """Tests loading of CASPDEBUG proteins."""
    scn.load(casp_version="debug")

def test_load_scnmin():
    """Tests loading of SCNMIN proteins."""
    scn.load(casp_version=12, casp_thinning="scnmin")

def test_load_scnunmin():
    """Tests loading of SCNMIN proteins."""
    scn.load(casp_version=12, casp_thinning="scnunmin")