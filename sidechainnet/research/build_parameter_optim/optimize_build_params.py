"""Load SC_HBUILD_INFO and optimize the corresponding build_params to minimize energy."""
import copy
import pickle
import numpy as np
import pkg_resources

from tqdm import tqdm
import torch
from sidechainnet.structure.build_info import BB_BUILD_INFO, SC_HBUILD_INFO
from sidechainnet.structure.fastbuild import get_all_atom_build_params
from sidechainnet.utils.openmm_loss import OpenMMEnergyH
from sidechainnet.dataloaders.SCNProtein import OPENMM_FORCEFIELDS, SCNProtein
from sidechainnet.tests.test_fastbuild import alphabet_protein


class BuildParamOptimizer(object):
    """Class to help optimize building parameters."""

    def __init__(self,
                 protein,
                 opt_bond_lengths=True,
                 opt_thetas=True,
                 opt_chis=True,
                 ffname=OPENMM_FORCEFIELDS):
        self.build_params = get_all_atom_build_params(SC_HBUILD_INFO, BB_BUILD_INFO)
        self._starting_build_params = copy.deepcopy(self.build_params)
        self.protein = self.prepare_protein(protein)
        self.ffname = ffname
        assert ffname == ['amber14/protein.ff15ipq.xml', 'amber14/spce.xml']
        self.energy_loss = OpenMMEnergyH()
        self.keys_to_optimize = []
        if opt_bond_lengths:
            self.keys_to_optimize.append('bond_lengths')
        if opt_thetas:
            self.keys_to_optimize.extend(['cthetas', 'sthetas'])
        if opt_chis:
            self.keys_to_optimize.extend(['cchis', 'schis'])
        self.params = self.create_param_list_from_build_params(self.build_params)
        self.losses = []

    def prepare_protein(self, protein: SCNProtein):
        """Prepare protein for optimization by building hydrogens/init OpenMM."""
        protein.sb = None
        protein.angles.requires_grad_()
        protein.fastbuild(add_hydrogens=True,
                          build_params=self.build_params,
                          inplace=True)
        protein.initialize_openmm(nonbonded_interactions=False)
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

    def save_build_params(self, path):
        """Write out build_params dict to path as pickle object."""
        with open(path, "wb") as f:
            pickle.dump(self.build_params, f)

    def optimize(self, opt='LBFGS', lr=1e-5, steps=100):
        """Optimize self.build_params to minimize OpenMMEnergyH."""
        to_optim = self.params
        p = self.protein

        self.build_params_history = [copy.deepcopy(self.params)]

        # LBFGS Loop
        if opt == 'LBFGS':
            # TODO Fails to optimize
            self.opt = torch.optim.LBFGS(to_optim, lr=lr)
            for i in tqdm(range(steps)):
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
            self.opt = torch.optim.SGD(to_optim, lr=lr)
            pbar = tqdm(range(steps), dynamic_ncols=True)
            best_loss = None
            counter = 0
            patience = 10
            epsilon = 1e-4
            for i in pbar:
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
                if (best_loss is None or
                    (lossnp < best_loss and np.abs(best_loss - lossnp) > epsilon)):
                    best_loss = lossnp
                    counter = 0
                elif counter > patience:
                    print("Stopping early.")
                    break
                elif counter < patience:
                    counter += 1
                self.losses.append(lossnp)
                self.opt.step()

                pbar.set_postfix({'loss': f"{lossnp:.2f}"})

        self.update_complete_build_params_with_optimized_params(to_optim)
        return self.build_params


def main():
    """Minimize the build parameters for an example alphabet protein."""
    p = alphabet_protein()
    bpo = BuildParamOptimizer(p,
                              opt_bond_lengths=True,
                              opt_thetas=True,
                              opt_chis=True,
                              ffname=OPENMM_FORCEFIELDS)
    build_params = bpo.optimize(opt='SGD', lr=1e-6, steps=10000)
    fn = pkg_resources.resource_filename("sidechainnet", "resources/build_params.pkl")
    with open(fn, "wb") as f:
        pickle.dump(build_params, f)


if __name__ == "__main__":
    main()
