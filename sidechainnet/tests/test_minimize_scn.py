"""Test functionality of minimizing SidechainNet."""

import sidechainnet as scn
from sidechainnet.create import create
from sidechainnet.utils.download import process_id
from sidechainnet.research.minimize_scn import do_pickle, process_index
from sidechainnet.examples import get_alphabet_protein
from sidechainnet.research.minimizer import SCNMinimizer


def test_minimize_alpha():
    protein = get_alphabet_protein()
    print(f"Minimizing Protein {protein.id}.")
    m = SCNMinimizer()
    m.minimize_scnprotein(protein, optimizer='lbfgs', epochs=10, path="./test_minimize_alpha")
