"""Test functionality of minimizing SidechainNet."""

import sidechainnet as scn
from sidechainnet.create import create
from sidechainnet.utils.download import process_id
from sidechainnet.utils.minimize_scn import do_pickle, process_index


def test_minimize_10p1JB0_12_X():
    unmin_path = "/net/pulsar/home/koes/jok120/scnmin220905/unmin"
    min_path = "/net/pulsar/home/koes/jok120/scnmin220905/min"

    # process_index(0, unmin_path, min_path)
    process_index(1, unmin_path, min_path)
    # process_index(2, unmin_path, min_path)


def test_process_raw_10p1JB0_12_X():
    create(12, 100, './tmp1JBO')


def test_better_generate():
    from sidechainnet.create import generate_all_from_proteinnet

    generate_all_from_proteinnet(
        proteinnet_dir="/net/pulsar/home/koes/jok120/proteinnet/data",
        sidechainnet_out='/net/pulsar/home/koes/jok120/scn220905',
        num_cores=16,
        limit=200,
        regenerate_scdata=True)


if __name__ == "__main__":
    test_minimize_10p1JB0_12_X()
    # test_process_raw_10p1JB0_12_X()
    # test_better_generate()