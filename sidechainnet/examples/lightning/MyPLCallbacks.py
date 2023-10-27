"""Custom PyTorch Lightning Callbacks for Convenience."""

import copy
import os
import sys
import traceback
import pytorch_lightning as pl
import torch
import wandb

from sidechainnet.dataloaders.SCNProtein import SCNProtein
from sidechainnet.structure.structure import inverse_trig_transform


class ResetOptimizersOnGlobalStep(pl.Callback):
    """Re-init the optimizer at the specified global step. Helps to handle new loss fn."""

    def __init__(self, on_step):
        super().__init__()
        self.on_step = on_step

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Reinitialize the pre-configured optimizers at self.on_step."""
        if pl_module.global_step == self.on_step:
            print("Reinitializing optimizers.")
            # print("Switching to SGD.")
            # pl_module.hparams.opt_lr = 1e-6
            # pl_module.hparams.opt_lr_scheduling = 'plateau'
            opt_dict = pl_module.configure_optimizers()
            trainer.optimizers = [opt_dict['optimizer']]
            if 'lr_scheduler' in opt_dict:
                trainer.lr_schedulers = [opt_dict['lr_scheduler']]


class VisualizeStructuresEveryNSteps(pl.Callback):
    """Perform structure visualization every N steps."""

    def __init__(self, n_steps):
        super().__init__()
        self.n_steps = n_steps

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Generate protein structure visualization only every self.n_steps."""
        if pl_module.global_step % self.n_steps == 0:
            self._generate_viz_with_pred_helper(outputs["pred_helper"], pl_module)

    def _generate_viz_with_pred_helper(self, pred_helper, pl_module):
        import pymol
        # Select the first protein in the batch to visualize
        p = pred_helper.batch_pred[0]
        p.numpy()
        pdbfile = os.path.join(pl_module.save_dir, "pdbs",
                               f"{pl_module.global_step:07}_{p.id}_pred.pdb")
        # Note that we must reshape the 2dim coord tensors back into 3dim for viz. The
        # prediction helper flattens coords for easier comparison with loss functions.
        p.coords = p.coords.reshape(len(p), -1, 3)
        p.hcoords = p.hcoords.reshape(len(p), -1, 3)
        p.to_pdb(pdbfile)
        wandb.log({"structures/train/molecule": wandb.Molecule(pdbfile)})
        p.coords = p.coords.reshape(-1, 3)
        p.hcoords = p.hcoords.reshape(-1, 3)

        ptrue = pred_helper.batch_true[0]
        ptrue_pdbfile = os.path.join(pl_module.save_dir, "pdbs",
                                     f"{pl_module.global_step:07}_{p.id}_true.pdb")
        try:
            ptrue.hcoords = ptrue.hcoords.reshape(len(p), -1, 3)
            ptrue.coords = ptrue.coords.reshape(len(p), -1, 3)
            ptrue.to_pdb(ptrue_pdbfile)
            ptrue.hcoords = ptrue.hcoords.reshape(-1, 3)
            ptrue.coords = ptrue.coords.reshape(-1, 3)
        except ValueError as e:
            print(ptrue, "failed to save to a PDB file.")
            print(ptrue.seq)
            print(ptrue.coords.shape, ptrue.coords)
            print(ptrue.hcoords.shape, ptrue.hcoords)
            traceback.print_exc()
            exit(1)

        # Now, open the two files in pymol, align them, show sidechains, and save PNG
        pymol.cmd.load(ptrue_pdbfile, "true")
        pymol.cmd.load(pdbfile, "pred")
        pymol.cmd.color("marine", "true")
        pymol.cmd.color("oxygen", "pred")
        rmsd, _, _, _, _, _, _ = pymol.cmd.align("true and name n+ca+c+o",
                                                 "pred and name n+ca+c+o",
                                                 quiet=True)
        pymol.cmd.zoom()
        pymol.cmd.show("lines", "not (name c,o,n and not pro/n)")
        pymol.cmd.hide("cartoon", "pred")
        both_png_path = os.path.join(pl_module.save_dir, "pngs",
                                     f"{pl_module.global_step:07}_{p.id}_both.png")
        # TODO: Ray tracing occurs despite ray=0
        with Suppressor():
            pymol.cmd.png(both_png_path, width=1000, height=1000, quiet=1, dpi=300, ray=0)
        both_pse_path = os.path.join(pl_module.save_dir, "pdbs",
                                     f"{pl_module.global_step:07}_{p.id}_both.pse")
        pymol.cmd.save(both_pse_path)
        wandb.save(both_pse_path, base_path=pl_module.save_dir)
        pymol.cmd.delete("all")
        wandb.log({"structures/train/png": wandb.Image(both_png_path)})


class Suppressor(object):
    """Supresses stdout."""

    def __enter__(self):
        """Modify stdout."""
        devnull = open('/dev/null', 'w')
        self.oldstdout_fno = os.dup(sys.stdout.fileno())
        os.dup2(devnull.fileno(), 1)

    def __exit__(self, type, value, traceback):
        """Undo stdout modification."""
        os.dup2(self.oldstdout_fno, 1)

    def write(self, x):
        """Do nothing."""
        pass
