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


def test_generate_all():
    from sidechainnet.create import generate_all
    generate_all(num_cores=8, regenerate_scdata=True)


def test_generate_all_from_proteinnet():
    generate_all_from_proteinnet("/home/jok120/proteinnet/data/",
                                 "/home/jok120/scn220502",
                                 limit=None,
                                 regenerate_scdata=True)
