"""Load SC_HBUILD_INFO and optimize the corresponding build_params to minimize energy."""
import copy

from tqdm import tqdm
import torch
from sidechainnet.structure.build_info import BB_BUILD_INFO, SC_HBUILD_INFO
from sidechainnet.structure.fastbuild import get_all_atom_build_params
from sidechainnet.utils.openmm_loss import OpenMMEnergyH
from sidechainnet.dataloaders.SCNProtein import OPENMM_FORCEFIELDS, SCNProtein
from sidechainnet.tests.test_fastbuild import alphabet_protein


class BuildParamOptimizer(object):
    """Class to help optimize building parameters."""

    def __init__(self, protein, ffname=OPENMM_FORCEFIELDS):
        self.build_params = get_all_atom_build_params(SC_HBUILD_INFO, BB_BUILD_INFO)
        self._starting_build_params = copy.deepcopy(self.build_params)
        self.protein = self.prepare_protein(protein)
        self.ffname = ffname
        assert ffname == ['amber14/protein.ff15ipq.xml', 'amber14/spce.xml']
        self.energy_loss = OpenMMEnergyH()
        # self.keys_to_optimize = ['bond_lengths', 'cthetas', 'sthetas', 'cchis', 'schis']
        self.keys_to_optimize = ['cthetas', 'sthetas']
        self.params = self.create_param_list_from_build_params(self.build_params)

    def prepare_protein(self, protein: SCNProtein):
        """Prepare protein for optimization by building hydrogens/init OpenMM."""
        protein.sb = None
        protein.angles.requires_grad_()
        protein.fastbuild(add_hydrogens=True,
                          build_params=self.build_params,
                          inplace=True)
        protein.initialize_openmm()
        return protein

    def create_param_list_from_build_params(self, build_params):
        """Extract optimizable parameters from full build_params dictionary."""
        params = []
        for root_atom in ['N', 'CA', 'C']:
            for param_key in self.keys_to_optimize:
                build_params[root_atom][param_key].requires_grad_()
                params.append(build_params[root_atom][param_key])
        return params

    def update_complete_build_params_with_optimized_params(self, optimized_params):
        """Update fill build_params dictionary with the optimized subset."""
        i = 0
        for root_atom in ['N', 'CA', 'C']:
            for param_key in self.keys_to_optimize:
                self.build_params[root_atom][param_key] = optimized_params[i]
                i += 1

    def optimize(self, opt='LBFGS'):
        """Optimize self.build_params to minimize OpenMMEnergyH."""
        to_optim = self.params
        p = self.protein

        self.losses = []
        self.build_params_history = [copy.deepcopy(self.params)]

        # LBFGS Loop
        if opt == 'LBFGS':
            # TODO Fails to optimize
            self.opt = torch.optim.LBFGS(to_optim, lr=1e-5)
            for i in tqdm(range(50)):
                # Note keeping
                self.build_params_history.append(
                    [copy.deepcopy(p.detach().cpu()) for p in to_optim])

                def closure():
                    self.opt.zero_grad()
                    # Update the build_params complete dict with the optimized values
                    self.update_complete_build_params_with_optimized_params(to_optim)
                    # Rebuild the protein
                    p.fastbuild(build_params=self.build_params,
                                add_hydrogens=True,
                                inplace=True)
                    loss = self.energy_loss.apply(p, p.hcoords)
                    loss.backward()
                    loss_np = float(loss.detach().numpy())
                    p._last_loss = loss_np
                    self.losses.append(loss_np)
                    return loss

                self.opt.step(closure)
                print(p._last_loss)

        # SGD Loop
        elif opt == 'SGD':
            self.opt = torch.optim.SGD(to_optim, lr=1e-10)
            for i in tqdm(range(100)):
                # Note keeping
                self.build_params_history.append(
                    [copy.deepcopy(p.detach().cpu()) for p in to_optim])
                self.opt.zero_grad()

                # Update the build_params complete dict with the optimized values
                self.update_complete_build_params_with_optimized_params(to_optim)

                # Rebuild the protein
                p.fastbuild(build_params=self.build_params,
                            add_hydrogens=True,
                            inplace=True)

                # Compute the new energy
                loss = self.energy_loss.apply(p, p.hcoords)
                loss.backward()
                lossnp = float(loss.detach().cpu().numpy())
                self.losses.append(lossnp)
                self.opt.step()

        self.update_complete_build_params_with_optimized_params(to_optim)
        return self.build_params


def main():
    """Minimize the build parameters for an example alphabet protein."""
    bpo = BuildParamOptimizer(alphabet_protein(), ffname=OPENMM_FORCEFIELDS)
    new_build_params = bpo.optimize(opt='SGD')
    print(new_build_params)


if __name__ == "__main__":
    main()
