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

    def minimize_scnprotein(self, p, use_sgd=False, verbose=False, path=None):
        """Minimize a single SCNProtein object."""
        # p = self.preprocess_scnprotein(p)  # TODO support missing atoms
        p.torch()
        # p.cuda()
        original_angles = (p.angles).detach().clone()
        to_optim = (p.angles).detach().clone().requires_grad_(True)#.cuda()

        energy_loss = OpenMMEnergyH()
        if not use_sgd:
            opt = torch.optim.LBFGS(
                [to_optim],
                lr=1e-5,
                max_iter=20,  # Def 20
                max_eval=None,
                tolerance_grad=1e-07,
                tolerance_change=1e-04,
                history_size=200,
                line_search_fn="strong_wolfe")  # Def: None
        else:
            opt = torch.optim.SGD([to_optim], lr=1e-5)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,
                                                               'min',
                                                               patience=3,
                                                               verbose=verbose)

        losses = []
        p.angles = to_optim
        p.fastbuild(add_hydrogens=True, inplace=True)

        # Record the starting performance
        best_loss_so_far = energy_loss.apply(p, p.hcoords)
        best_angles_so_far = to_optim
        best_loss_updated = False
        epoch_last_best_loss_seen = -1

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
            if verbose:
                print(lossnp)
            return loss

        for i in range(10000):
            # Early Stopping and Loss Evaluation
            loss = energy_loss.apply(p, p.hcoords)
            if loss < best_loss_so_far and torch.abs(loss - best_loss_so_far) > 20:
                best_loss_so_far = loss
                best_angles_so_far = to_optim
                best_loss_updated = True
                epoch_last_best_loss_seen = i
                print("Loss has been updated. ", loss)
            elif i - epoch_last_best_loss_seen > 4:
                print(f"Stopping early after {i} epochs.", loss)
                break

            if use_sgd:
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
                opt.step()
            else:
                opt.step(closure)
                scheduler.step(loss)
                losses.append(float(loss.detach().cpu().numpy()))

        # if verbose:
        #     plt.show()

        if not best_loss_updated:
            print("The protein was not minimized correctly.")
            plt.savefig(path.replace(".pkl", ".png").replace("/min/", "/failed/"))
            raise ValueError("The protein was not minimized correctly.")

        # Plot the minimization performance
        sns.lineplot(data=losses)
        plt.title("Protein Potential Energy")
        plt.xlabel("Optimization Step")
        if path is not None:
            plt.savefig(path.replace(".pkl", ".png"))

        # Update final structure
        p.angles = best_angles_so_far
        p.fastbuild(add_hydrogens=True, inplace=True)
        p.numpy()

        return best_angles_so_far


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
