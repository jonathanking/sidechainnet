"""Minimizes SCNProteins."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from tqdm import tqdm

from sidechainnet.utils.openmm_loss import OpenMMEnergyH

torch.set_printoptions(sci_mode=False, precision=3)
np.set_printoptions(suppress=True)


class SCNMinimizer(object):

    def __init__(self, dataset=None) -> None:
        """Create a SCNMinimizer given a SCNDataset object."""
        self.data = dataset

    def minimize_dataset(self, verbose=False):
        """Minimize every protein in the dataset."""
        for p in tqdm(self.data):
            self.minimize_scnprotein(p, verbose=verbose)
        return self.data

    def preprocess_scnprotein(self, p):
        # Add missing atoms
        p.add_hydrogens()
        p.make_pdbfixer()
        p.pdbfixer.addMissingAtoms()
        p.pdbfixer.findMissingAtoms()

        return p

    def minimize_scnprotein(self, p, use_sgd=False, verbose=False):
        """Minimize a single SCNProtein object."""
        # p = self.preprocess_scnprotein(p)  # TODO support missing atoms
        p.torch()
        p.cuda()
        original_angles = (p.angles).detach().clone()
        to_optim = (p.angles).detach().clone().requires_grad_(True).cuda()

        energy_loss = OpenMMEnergyH()
        if not use_sgd:
            opt = torch.optim.LBFGS(
                [to_optim],
                lr=1,
                max_iter=100,  # Def 20
                max_eval=None,
                tolerance_grad=1e-07,
                tolerance_change=1e-04,
                history_size=100,
                line_search_fn="strong_wolfe")  # Def: None
        else:
            opt = torch.optim.SGD([to_optim], lr=1e-5)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,
                                                               'min',
                                                               patience=8,
                                                               verbose=verbose)

        losses = []

        def turned_nan(original, current):
            return torch.isnan(current).sum() > torch.isnan(original).sum()

        def closure():
            if turned_nan(original_angles, to_optim):
                raise ValueError

            if torch.is_grad_enabled():
                opt.zero_grad()

            # Rebuild the coordinates from the angles
            p.angles = to_optim
            p.fastbuild(add_hydrogens=True, inplace=True)

            # Compute the loss on the coordinates
            loss = energy_loss.apply(p, p.hcoords)

            # Backprop to angles
            if loss.requires_grad:
                loss.backward()

            torch.nn.utils.clip_grad_value_(to_optim, 1)
            lossnp = float(loss.detach().cpu().numpy())
            losses.append(lossnp)
            if verbose:
                print(lossnp)
            scheduler.step(loss)
            return loss

        for i in range(10):
            opt.step(closure)

        sns.lineplot(data=losses)
        plt.title("Protein Potential Energy")
        plt.xlabel("Optimization Step")
        # if verbose:
        #     plt.show()

        # Update final structure
        p.add_hydrogens(from_angles=True, angles=to_optim)
        p.angles = to_optim.detach().cpu().numpy()
        p.hcoords = p.coords = p.hcoords.detach().cpu().numpy()

        return to_optim


if __name__ == "__main__":
    import sidechainnet as scn
    from sidechainnet.utils.minimizer import SCNMinimizer
    d = scn.load("debug",
                 scn_dataset=True,
                 complete_structures_only=True,
                 trim_edges=True,
                 scn_dir="/home/jok120/sidechainnet_data",
                 filter_by_resolution=False)
    m = SCNMinimizer(d)
    p = d[0]
    m.minimize_scnprotein(p, verbose=True)
    p.hcoords
