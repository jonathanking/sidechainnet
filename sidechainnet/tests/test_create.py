import sidechainnet as scn
import pytest


def test_create_custom():
    with open("/home/jok120/sidechainnet/data/proteinnet/training_100_ids.txt", "r") as f:
        training_ids = f.read().splitlines()
    scn.create_custom(training_ids[:10], "custom01.pkl", "first 5 entries of CASP12",
                      "/home/jok120/sidechainnet/data/proteinnet")

def test_get_proteinnet_ids():
    ids = scn.get_proteinnet_ids(12, "valid-10", 100)

