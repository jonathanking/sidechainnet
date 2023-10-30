from sidechainnet.create import make_unmodified_seq_entry, generate_all_from_proteinnet
import sidechainnet as scn
import pytest


def test_get_proteinnet_ids():
    ids = scn.get_proteinnet_ids(12, "valid-10", 100)


def test_make_unmodified_seq_entry():
    pn_seq = "CCCCC"
    unmod_seq = ["CYS", "CYS", "CYS", "CYS", "CYM"]
    mask = "+++++"
    result = make_unmodified_seq_entry(pn_seq, unmod_seq, mask)
    assert result == "CYS CYS CYS CYS CYM"

    pn_seq = "CCCCC"
    unmod_seq = ["CYS", "CYS", "CYS", "CYS"]
    mask = "++++-"
    result = make_unmodified_seq_entry(pn_seq, unmod_seq, mask)
    assert result == "CYS CYS CYS CYS CYS"

    pn_seq = "CCCCC"
    unmod_seq = ["CYS", "CYS", "CYM", "CYS", "CYS"]
    mask = "+++++"
    result = make_unmodified_seq_entry(pn_seq, unmod_seq, mask)
    assert result == "CYS CYS CYM CYS CYS"
