from sidechainnet.create import make_unmodified_seq_entry
import sidechainnet as scn
import pytest


def test_create_custom():
    with open("/home/jok120/sidechainnet/data/proteinnet/training_100_ids.txt", "r") as f:
        training_ids = f.read().splitlines()
    scn.create_custom(training_ids[:10], "custom01.pkl", "first 5 entries of CASP12",
                      "/home/jok120/sidechainnet/data/proteinnet")

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

def test_modified_residues_are_reported_in_scn_entry_ums():
    pnids = ['20#2FZT_1_A', "70#2E87_1_A", "20#3S0X_1_A", '70#2E87_1_A']
