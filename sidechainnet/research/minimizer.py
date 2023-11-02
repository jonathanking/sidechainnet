"""Minimizes SCNProteins."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from tqdm import tqdm
from random import random

from sidechainnet.utils.openmm_loss import OpenMMEnergyH

torch.set_printoptions(sci_mode=False, precision=3)
np.set_printoptions(suppress=True)

CLIP_GRAD_VAL = 1e-4


class SCNMinimizer(object):
    """Minimize a SCNDataset object."""
    def __init__(self, dataset=None, verbose=True) -> None:
        """Create a SCNMinimizer given a SCNDataset object."""
        self.data = dataset
        self.verbose = verbose

    def minimize_dataset(self, verbose=False):
        """Minimize every protein in the dataset."""
        for p in tqdm(self.data):
            self.minimize_scnprotein(p, verbose=verbose)
        return self.data

    def preprocess_scnprotein(self, p):
        """Preprocess a protein object for minimization."""
        # Add missing atoms
        p.add_hydrogens()
        p._make_pdbfixer()
        p.pdbfixer.addMissingAtoms()
        p.pdbfixer.findMissingAtoms()

        return p

    def _get_optimizer(self, to_optim):
        if self.optimizer_name == "lbfgs":
            opt = torch.optim.LBFGS(
                [to_optim],
                lr=self.lr,
                max_iter=self.max_iter,  # Def 20
                max_eval=self.max_eval,  # Def 1.5 * max_iter
                tolerance_grad=1e-07,
                tolerance_change=1e-06,
                history_size=150,
                line_search_fn="strong_wolfe")
        elif self.optimizer_name == "sgd":
            opt = torch.optim.SGD([to_optim], lr=self.lr)
        elif self.optimizer_name == "adam":
            opt = torch.optim.Adam([to_optim], lr=self.lr)
        else:
            raise ValueError(f"Optimizer {self.optimizer_name} not supported.")
        return opt

    def minimize_scnprotein(
        self,
        p,
        *,
        optimizer="lbfgs",
        path="./",
        starting_lr=1e-4,
        max_iter=20,
        epochs=10_000,
        lr_decay=True,
        max_eval=None,
        patience=10,
        minimum_lr=1e-6,
        record_structure_every_n=None,
        suffix="",
        grad_clip=None,
    ):
        """Minimize a single SCNProtein object."""
        # Record input config
        self.optimizer_name = optimizer
        self.lr = starting_lr
        self.max_iter = max_iter
        self.epochs = epochs
        self.lr_decay = lr_decay
        self.max_eval = max_eval
        self.patience = patience
        self.record_structure_every_n = record_structure_every_n
        self.suffix = suffix
        self.grad_clip = CLIP_GRAD_VAL if grad_clip is None else grad_clip
        p.numpy()
        starting_angles = np.copy(p.angles)

        loss_curves = []
        improved = False

        self.vprint("Starting minimization...")
        while self.lr > minimum_lr and not improved:
            self.vprint(f"   LR = {self.lr}")
            min_results = self._minimize_scnprotein(p)
            improved = (min_results["best_loss_updated"]
                        and min_results["loss"] < min_results["starting_loss"])
            loss_curves.append(min_results["losses"])

            if not improved and self.lr / 10 > minimum_lr:
                self.lr /= 10
                self.vprint(f"The protein was not minimized correctly. Lowering LR  to "
                            f"{self.lr} and trying again.")
                p.angles = starting_angles

        if not improved:
            self.vprint("The protein was not minimized correctly.")
            # plt.savefig(path.replace(".pkl", ".png").replace("/min/", "/failed/"))
            raise ValueError("The protein was not minimized correctly.")
        else:
            p.angles = min_results["best_angles_so_far"]
            p.fastbuild(add_hydrogens=True, inplace=True)
            p.numpy()
            self.vprint("Updated protein angles and hcoords.")

        self.vprint("Minimization complete.")

        # Plot all loss curves
        plt.figure(figsize=(10, 10))
        for i, curve in enumerate(loss_curves[-1:]):
            plt.plot(curve, label=f"Curve {i}")
            plt.legend()
            plt.title("Protein Potential Energy")
            plt.xlabel("Optimization Step")

            if path is not None:
                plt.savefig(path.replace(".pkl", ".png").replace(".png", f"loss{i}.png"))

        return p

    def _minimize_scnprotein(self, p):
        """Minimize a single SCNProtein object."""
        # TODO support missing atoms
        min_results = {}
        p.torch()
        to_optim = (p.angles).detach().clone().requires_grad_(True)
        opt = self._get_optimizer(to_optim)
        energy_loss = OpenMMEnergyH()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,
                                                               'min',
                                                               patience=self.patience //
                                                               2,
                                                               verbose=self.verbose,
                                                               threshold=5,
                                                               threshold_mode='abs')

        # Prepare for minimization
        p.angles = to_optim
        p.fastbuild(add_hydrogens=True, inplace=True)
        starting_angles = (p.angles).detach().clone()

        best_loss_so_far = energy_loss.apply(p, p.hcoords)
        starting_loss = float(best_loss_so_far.detach().cpu())
        best_angles_so_far = to_optim.detach().clone()
        best_loss_updated = False
        epoch_last_best_loss_seen = -1

        stopped_early = False
        loss = starting_loss
        losses = [starting_loss]
        self.vprint(f"Starting loss = {loss:.2f}")

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

            torch.nn.utils.clip_grad_value_(to_optim, self.grad_clip)

            return loss

        for i in range(self.epochs):

            if (self.record_structure_every_n is not None
                    and i % self.record_structure_every_n == 0):
                # save with name padded left with 0s
                p.pickle(f"{p.id}_step_{i:05d}{self.suffix}.pkl")
                p.to_pdb(f"{p.id}_step_{i:05d}{self.suffix}.pdb")
                # p.to_png(f"{p.id}_step_{i:05d}.png")

            if self.optimizer_name != "lbfgs":
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
                torch.nn.utils.clip_grad_value_(to_optim, self.grad_clip)
                opt.step()
            else:
                opt.step(closure)
                # Early Stopping and Loss Evaluation
                loss = energy_loss.apply(p, p.hcoords)

            lossnp = float(loss.detach().cpu().numpy())
            losses.append(lossnp)
            self.vprint(f"Epoch {i}:\t{lossnp:.2f}")

            if self.lr_decay:
                scheduler.step(loss)

            if loss < best_loss_so_far:
                best_loss_so_far = loss
                best_angles_so_far = to_optim.detach().clone()
                best_loss_updated = True
                epoch_last_best_loss_seen = i
                self.vprint("Updated best angles.")
            elif i - epoch_last_best_loss_seen > self.patience:
                stopped_early = True
                break

        loss = energy_loss.apply(p, p.hcoords)
        min_results["losses"] = losses
        min_results["best_loss_updated"] = best_loss_updated
        min_results["loss"] = float(loss.detach().cpu())
        min_results["starting_loss"] = starting_loss
        min_results["best_angles_so_far"] = best_angles_so_far.detach().clone()
        min_results["best_loss_so_far"] = float(best_loss_so_far.detach().cpu())

        self.vprint(f"Final loss = {loss:.2f}")

        return min_results

    def vprint(self, msg):
        if self.verbose:
            print(msg)


if __name__ == "__main__":
    import sidechainnet as scn
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
