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
            self._generate_structure_viz(batch, outputs['sc_angs_pred'], split='train')

    def _generate_structure_viz(self, batch, sc_angs_pred, split):
        import pymol

        sc_angs_pred_rad = inverse_trig_transform(sc_angs_pred, n_angles=6)
        # Select the first protein in the batch to visualize
        j = -1
        b = batch[j]
        if torch.is_tensor(b.coords):
            p = SCNProtein(
                coordinates=b.coords.detach().cpu().numpy(),
                angles=b.angles.detach().cpu().numpy(),
                sequence=b.seq,
                unmodified_seq=b.unmodified_seq,
                mask=b.mask,
                evolutionary=b.evolutionary,
                secondary_structure=b.secondary_structure,
                resolution=b.resolution,
                is_modified=b.is_modified,
                id=b.id,
                split=b.split,
                add_sos_eos=b.add_sos_eos,
            )
            p.numpy()
        else:
            p = copy.deepcopy(batch[j])
        assert p is not batch[j]
        p.trim_to_max_seq_len()
        p.angles[:, 6:] = sc_angs_pred_rad[j, 0:len(p)].detach().cpu().numpy()
        p.numpy()
        p.build_coords_from_angles()  # Make sure the coordinates saved to PDB are updated
        pdbfile = os.path.join(self.save_dir, "pdbs", f"{p.id}_pred.pdb")
        p.to_pdb(pdbfile)
        wandb.log({f"structures/{split}/molecule": wandb.Molecule(pdbfile)})

        # Now to generate the files for the true structure
        ptrue = batch[j]
        ptrue_pdbfile = os.path.join(self.save_dir, "pdbs", f"{p.id}_true.pdb")
        try:
            ptrue.to_pdb(ptrue_pdbfile)
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
        both_png_path = os.path.join(self.save_dir, "pngs", f"{p.id}_both.png")
        # TODO: Ray tracing occurs despite ray=0
        with Suppressor():
            pymol.cmd.png(both_png_path, width=1000, height=1000, quiet=1, dpi=300, ray=0)
        both_pse_path = os.path.join(self.save_dir, "pdbs", f"{p.id}_both.pse")
        pymol.cmd.save(both_pse_path)
        wandb.save(both_pse_path, base_path=self.save_dir)
        pymol.cmd.delete("all")
        wandb.log({f"structures/{split}/png": wandb.Image(both_png_path)})


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
