from sidechainnet.examples import alphabet_protein
from sidechainnet.research.build_parameter_optim.optimize_build_params import BuildParamOptimizer


def test_other_protein():

    p = alphabet_protein()
    p2 = alphabet_protein()

    bpo = BuildParamOptimizer(p)
    bpo.optimize(opt='SGD')

    p.fastbuild(add_hydrogens=True, build_params=bpo.build_params, inplace=True)
    p.to_3Dmol(other_protein=p2)

