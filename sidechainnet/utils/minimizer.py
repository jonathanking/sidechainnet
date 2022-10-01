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

    def minimize_scnprotein(self,
                            p,
                            use_sgd=False,
                            verbose=False,
                            path=None,
                            starting_lr=1e-4,
                            max_iter=20,
                            epochs=10_000,
                            lr_decay=True,
                            max_eval=None,
                            patience=10):
        """Minimize a single SCNProtein object."""
        # p = self.preprocess_scnprotein(p)  # TODO support missing atoms
        p.torch()
        # p.cuda()
        to_optim = (p.angles).detach().clone().requires_grad_(True)  #.cuda()

        energy_loss = OpenMMEnergyH()
        if not use_sgd:
            opt = torch.optim.LBFGS(
                [to_optim],
                lr=starting_lr,
                max_iter=max_iter,  # Def 20
                max_eval=max_eval,  # Def 1.5 * max_iter
                tolerance_grad=1e-07,
                tolerance_change=1e-02,
                history_size=150,
                line_search_fn="strong_wolfe")  # Def: None
        else:
            opt = torch.optim.SGD([to_optim], lr=1e-5)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,
                                                               'min',
                                                               patience=patience,
                                                               verbose=verbose,
                                                               threshold=5,
                                                               threshold_mode='abs')

        losses = []
        p.angles = to_optim
        p.fastbuild(add_hydrogens=True, inplace=True)

        # Record the starting performance
        best_loss_so_far = energy_loss.apply(p, p.hcoords)
        starting_loss = float(best_loss_so_far.detach().cpu())
        best_angles_so_far = to_optim
        best_loss_updated = False
        epoch_last_best_loss_seen = -1
        starting_angles = (p.angles).detach().clone()
        stopped_early = False
        loss = starting_loss
        losses.append(starting_loss)
        print(f"Starting loss = {loss:.2f}")

        def turned_nan(original, current):
            return torch.isnan(current).sum() > torch.isnan(original).sum()

        def closure():
            if turned_nan(starting_angles, to_optim):
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

            return loss

        for i in range(epochs):

            if use_sgd:
                if turned_nan(starting_angles, to_optim):
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
                # Early Stopping and Loss Evaluation
                loss = energy_loss.apply(p, p.hcoords)
                losses.append(float(loss.detach().cpu().numpy()))
                print(f"Epoch {i}:\t{loss:.2f}")
                if lr_decay:
                    scheduler.step(loss)

                if loss < best_loss_so_far and torch.abs(loss - best_loss_so_far) > 10:
                    best_loss_so_far = loss
                    best_angles_so_far = to_optim
                    best_loss_updated = True
                    epoch_last_best_loss_seen = i
                    print(f"Epoch {i}: Loss has been updated. ", loss)
                elif i - epoch_last_best_loss_seen > patience:
                    print(f"Stopping early after {i} epochs.", loss)
                    stopped_early = True
                    break
                # elif loss > starting_loss + 100 or loss > prev_loss:
                elif loss > starting_loss + 100:
                    break

        if stopped_early:
            # then we're done
            pass
        elif ((not best_loss_updated and epochs > 1) or
              loss > starting_loss) and starting_lr < 1e-6:
            print("The protein was not minimized correctly.")
            plt.savefig(path.replace(".pkl", ".png").replace("/min/", "/failed/"))
            raise ValueError("The protein was not minimized correctly.")
        elif ((not best_loss_updated and epochs > 1) or loss > starting_loss):
            print(f"The protein was not minimized correctly. Lowering LR  to "
                  f"{starting_lr/10} and trying again.")
            p.angles = starting_angles
            new_loss, better_angles = self.minimize_scnprotein(p,
                                                               use_sgd=use_sgd,
                                                               starting_lr=starting_lr /
                                                               10,
                                                               verbose=verbose,
                                                               path=path,
                                                               max_iter=max_iter,
                                                               epochs=epochs,
                                                               lr_decay=lr_decay,
                                                               max_eval=max_eval)
            if new_loss < best_loss_so_far:
                best_angles_so_far = better_angles

        # Plot the minimization performance
        sns.lineplot(data=losses)
        plt.title("Protein Potential Energy")
        plt.xlabel("Optimization Step")

        if path is not None:
            plt.savefig(path.replace(".pkl", ".png"))

        # Update final structure
        p.angles = best_angles_so_far
        p.fastbuild(add_hydrogens=True, inplace=True)
        loss = energy_loss.apply(p, p.hcoords)
        print(f"Final loss = {loss:.2f}")
        p.numpy()

        return best_loss_so_far, best_angles_so_far


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
