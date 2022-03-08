import sys
from typing import Dict
import pymol
import copy
import os
import numpy as np
import torch
from sidechainnet.dataloaders.SCNProtein import SCNProtein
from sidechainnet.examples.losses import angle_mse, angle_diff
from sidechainnet.structure.build_info import ANGLE_IDX_TO_NAME_MAP
from sidechainnet.structure.structure import inverse_trig_transform
from sidechainnet.utils.openmm import OpenMMEnergyH
from sidechainnet.examples.optim import ScheduledOptim

import pytorch_lightning as pl

from sidechainnet.utils.sequence import VOCAB
import sidechainnet as scn

import wandb


class LitSidechainTransformer(pl.LightningModule):
    """PyTorch Lightning module for SidechainTransformer."""

    @staticmethod
    def add_model_specific_args(parent_parser):
        """Add model specific arguments to a global parser."""

        def my_bool(s):
            """Allow bools instead of using pos/neg flags."""
            return s != 'False'

        model_args = parent_parser.add_argument_group("LitSidechainTransformer")
        model_args.add_argument('--d_seq_embedding',
                                '-dse',
                                type=int,
                                default=512,
                                help="Dimension of sequence embedding.")
        model_args.add_argument('--d_nonseq_data',
                                '-dnsd',
                                type=int,
                                default=35,
                                help="Dimension of non-sequence input embedding.")
        model_args.add_argument('--d_out',
                                '-do',
                                type=int,
                                default=12,
                                help="Dimension of desired model output.")
        model_args.add_argument('--d_in',
                                '-di',
                                type=int,
                                default=256,
                                help="Dimension of desired transformer model input.")
        model_args.add_argument('--d_feedforward',
                                '-dff',
                                type=int,
                                default=2048,
                                help="Dimmension of the inner layer of the feed-forward "
                                "layer at the end of every Transformer block.")
        model_args.add_argument('--n_heads',
                                '-nh',
                                type=int,
                                default=8,
                                help="Number of attention heads.")
        model_args.add_argument('--n_layers',
                                '-nl',
                                type=int,
                                default=6,
                                help="Number of layers in each the encoder/decoder "
                                "(if present).")
        model_args.add_argument("--embed_sequence",
                                type=my_bool,
                                default="True",
                                help="Whether or not to use embedding layer in the "
                                "transformer model.")
        model_args.add_argument("--transformer_activation",
                                type=str,
                                default="relu",
                                help="Activation for Transformer layers.")
        model_args.add_argument("--log_structures",
                                type=my_bool,
                                default="True",
                                help="Whether or not to log structures while training.")

        return parent_parser

    def __init__(
            self,
            # Model specific args from CLI
            d_seq_embedding=20,
            d_nonseq_data=35,  # 5 bb, 21 PSSM, 8 ss
            d_in=256,
            d_out=6,
            d_feedforward=1024,
            n_heads=8,
            n_layers=1,
            embed_sequence=True,
            transformer_activation='relu',
            # Shared arguments from CLI
            loss_name='mse',
            opt_name='adam',
            opt_lr=1e-2,
            opt_lr_scheduling='plateau',
            opt_lr_scheduling_metric='val_loss',
            opt_patience=5,
            opt_min_delta=0.01,
            opt_weight_decay=1e-5,
            opt_n_warmup_steps=5_000,
            dropout=0.1,
            # Other
            dataloader_name_mapping=None,
            angle_means=None,
            **kwargs):
        """Create a LitSidechainTransformer module."""
        super().__init__()
        self.save_hyperparameters()

        # Initialize layers
        if embed_sequence:
            self.input_embedding = torch.nn.Embedding(len(VOCAB),
                                                      d_seq_embedding,
                                                      padding_idx=VOCAB.pad_id)
        self.ff1 = torch.nn.Linear(d_nonseq_data + d_seq_embedding, d_in)
        self.encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_in,
            nhead=n_heads,
            dim_feedforward=d_feedforward,
            dropout=dropout,
            activation=transformer_activation,
            batch_first=True,
            norm_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(self.encoder_layer,
                                                               num_layers=n_layers)
        self.ff2 = torch.nn.Linear(d_in, d_out)
        self.output_activation = torch.nn.Tanh()

        # Initialize model parameters
        self._init_parameters()
        if angle_means is not None:
            self._init_angle_mean_projection()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def _init_angle_mean_projection(self):
        """Initialize last projection bias s.t. model starts out predicting the mean."""
        angle_means = np.arctanh(self.hparams.angle_means)
        self.ff2.bias = torch.nn.Parameter(angle_means)
        torch.nn.init.zeros_(self.ff2.weight)
        self.ff2.bias.requires_grad_ = False

    def _get_seq_pad_mask(self, seq):
        # Seq is Batch x L
        assert len(seq.shape) == 2
        return seq == VOCAB.pad_id

    def forward(self, x, seq):
        """Run one forward step of the model."""
        seq = seq.to(self.device)
        padding_mask = self._get_seq_pad_mask(seq)
        if self.hparams.embed_sequence:
            seq = self.input_embedding(seq)
        x = torch.cat([x, seq], dim=-1)
        x = self.ff1(x)
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        x = self.ff2(x)
        x = self.output_activation(x)
        return x

    # Lightning Hooks

    def configure_optimizers(self):
        """Prepare optimizer and schedulers.

        Args:
            optimizer (str): Name of optimizer ('adam', 'sgd')
            learning_rate (float): Learning rate for optimizer.
            weight_decay (bool, optional): Use optimizer weight decay. Defaults to True.

        Returns:
            dict: Pytorch Lightning dictionary with keys "optimizer" and "lr_scheduler".
        """
        # Prepare optimizer
        if self.hparams.opt_name == "adam":
            opt = torch.optim.Adam(filter(lambda x: x.requires_grad, self.parameters()),
                                   lr=self.hparams.opt_lr,
                                   weight_decay=self.hparams.opt_weight_decay)
        elif self.hparams.opt_name == "adamw":
            opt = torch.optim.AdamW(filter(lambda x: x.requires_grad, self.parameters()),
                                    lr=self.hparams.opt_lr,
                                    weight_decay=self.hparams.opt_weight_decay)
        elif self.hparams.opt_name == "sgd":
            opt = torch.optim.SGD(filter(lambda x: x.requires_grad, self.parameters()),
                                  lr=self.hparams.opt_lr,
                                  weight_decay=self.hparams.opt_weight_decay)

        # Prepare scheduler
        if self.hparams.opt_lr_scheduling == "noam":
            opt = ScheduledOptim(opt, self.hparams.d_in, self.hparams.opt_n_warmup_steps)
            sch = None
            # TODO: Use better warm up scheduler
        else:
            sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                patience=self.hparams.opt_patience,
                verbose=True,
                threshold=self.hparams.opt_min_delta,
                mode='min'
                if 'acc' not in self.hparams.opt_lr_scheduling_metric else 'max')

        d = {"optimizer": opt}
        if sch is not None:
            d["lr_scheduler"] = {
                "scheduler": sch,
                "interval": "epoch",
                "frequency": 1,
                "monitor": self.hparams.opt_lr_scheduling_metric,
                "strict": True,
                "name": None,
            }

        return d

    def _prepare_model_input(self, batch):
        # True values still have nans, replace with 0s so they can go into the network
        # Also select out backbone and sidechain angles
        bb_angs = torch.nan_to_num(batch.angles[:, :, :6], nan=0.0)
        sc_angs_true_untransformed = batch.angles[:, :, 6:]

        # Since *batches* are padded with 0s, we replace with nan for convenient loss fns
        sc_angs_true_untransformed[sc_angs_true_untransformed.eq(0).all(dim=-1)] = np.nan

        # Result after transform (6 angles, sin/cos): (B x L x 12)
        sc_angs_true = scn.structure.trig_transform(sc_angs_true_untransformed).reshape(
            sc_angs_true_untransformed.shape[0], sc_angs_true_untransformed.shape[1], 12)

        # Stack model inputs into a single tensor
        model_in = torch.cat([bb_angs, batch.secondary, batch.evolutionary], dim=-1)

        return model_in.to(self.device), sc_angs_true.to(self.device)

    def training_step(self, batch, batch_idx):
        """Perform a single step of training (model in, model out, log loss).

        Args:
            batch (List): List of Protein objects.
            batch_idx (int): Integer index of the batch.
        """
        model_in, sc_angs_true = self._prepare_model_input(batch)

        # Predict sidechain angles given input and sequence
        sc_angs_pred = self(model_in, batch.seqs_int)  # ( B x L x 12)

        # Compute loss and step
        loss_dict = self._get_losses(batch,
                                     sc_angs_true,
                                     sc_angs_pred,
                                     do_struct=batch_idx == 0,
                                     split='train')
        self.log(
            'losses/train/rmse',
            torch.sqrt(loss_dict['mse']),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        self.log("trainer/batch_size", float(len(batch)), on_step=True, on_epoch=False)
        self._log_angle_metrics(loss_dict, 'train')

        return loss_dict

    def validation_step(self,
                        batch,
                        batch_idx,
                        dataloader_idx=0) -> Dict[str, torch.Tensor]:
        """Single validation step with multiple possible DataLoaders."""
        model_in, sc_angs_true = self._prepare_model_input(batch)

        # Predict sidechain angles given input and sequence
        sc_angs_pred = self(model_in, batch.seqs_int)  # ( B x L x 12)

        # Compute loss
        loss_dict = self._get_losses(batch,
                                     sc_angs_true,
                                     sc_angs_pred,
                                     do_struct=batch_idx == 0 and dataloader_idx == 0,
                                     split='valid')

        name = f"losses/valid/{self.hparams.dataloader_name_mapping[dataloader_idx]}_rmse"
        self.log(name,
                 torch.sqrt(loss_dict['mse']),
                 on_step=False,
                 on_epoch=True,
                 prog_bar=name == self.hparams.opt_lr_scheduling_metric,
                 add_dataloader_idx=False)
        self._log_angle_metrics(loss_dict, 'valid',
                                self.hparams.dataloader_name_mapping[dataloader_idx])

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        """Single test step with multiple possible DataLoaders. Same as validation."""
        model_in, sc_angs_true = self._prepare_model_input(batch)

        # Predict sidechain angles given input and sequence
        sc_angs_pred = self(model_in, batch.seqs_int)  # ( B x L x 12)

        # Compute loss
        loss_dict = self._get_losses(batch,
                                     sc_angs_true,
                                     sc_angs_pred,
                                     do_struct=batch_idx == 0,
                                     split='test')
        self.log("losses/test/rmse",
                 torch.sqrt(loss_dict['mse']),
                 on_step=False,
                 on_epoch=True)
        self._log_angle_metrics(loss_dict, 'test')

    def _compute_angle_metrics(self,
                               sc_angs_true,
                               sc_angs_pred,
                               loss_dict,
                               acc_tol=np.deg2rad(20)):
        """Compute MAE(X1-5) and accuracy (correct within 20 deg).

        Args:
            sc_angs_true (torch.Tensor): True sidechain angles, padded with nans.
            sc_angs_pred (torch.Tensor): Predicted sidechain angles.
            loss_dict (dict): Dictionary for loss and metric record keeping.

        Returns:
            loss_dict (dict): Updated dictionary containing new keys MAE_X1...ACC.
        """
        # Conver sin/cos values into radians
        sc_angs_pred_rad = inverse_trig_transform(sc_angs_pred.detach(), n_angles=6)
        sc_angs_true_rad = inverse_trig_transform(sc_angs_true.detach(), n_angles=6)

        # Absolue Error Only (contains nans)
        abs_error = torch.abs(angle_diff(sc_angs_true_rad, sc_angs_pred_rad))

        loss_dict['angle_metrics'] = {}

        # Compute MAE by angle; Collapse sequence dimension and batch dimension
        mae_by_angle = torch.nanmean(abs_error, dim=1).mean(dim=0)
        assert len(mae_by_angle) == 6, "MAE by angle should have length 6."
        for i in range(6):
            angle_name_idx = i + 6
            angle_name = ANGLE_IDX_TO_NAME_MAP[angle_name_idx]
            cur_mae = mae_by_angle[i]
            if torch.isnan(cur_mae):
                continue
            loss_dict["angle_metrics"][f"{angle_name}_mae_rad"] = cur_mae
            loss_dict["angle_metrics"][f"{angle_name}_mae_deg"] = torch.rad2deg(cur_mae)

        # Compute angles within tolerance (contains nans)
        correct_angle_preds = abs_error < acc_tol

        # An entire residue is correct if the residue has >= 1 predicted angles
        # and all of those angles are correct. When computing correctness, an angle is
        # correct if it met the tolerance or if it is nan.
        res_is_predicted = ~torch.isnan(abs_error).all(dim=-1)  # Avoids all-nan rows
        res_is_correct = torch.logical_or(correct_angle_preds,
                                          torch.isnan(abs_error)).all(dim=-1)
        res_is_predicted_and_correct = torch.logical_and(res_is_predicted, res_is_correct)
        accuracy = res_is_predicted_and_correct.sum() / res_is_predicted.sum()

        loss_dict["angle_metrics"]["acc"] = accuracy

        # Compute Residue-Independent Accuracy (RIA) which evaluates each angle indp.
        ang_is_predicted = ~torch.isnan(correct_angle_preds)
        ang_is_correct = torch.logical_or(correct_angle_preds, torch.isnan(abs_error))
        num_correct_angles = torch.logical_and(ang_is_predicted, ang_is_correct).sum()
        ria = num_correct_angles / ang_is_predicted.sum()
        loss_dict['angle_metrics']['ria'] = ria

        return loss_dict

    def _log_angle_metrics(self, loss_dict, split, dataloader_name=""):
        dl_name_str = dataloader_name + "_" if dataloader_name else ""

        angle_metrics_dict = {
            f"metrics/{split}/{dl_name_str}{k}": loss_dict['angle_metrics'][k]
            for k in loss_dict['angle_metrics'].keys()
        }
        self.log_dict(angle_metrics_dict,
                      on_epoch=True,
                      on_step=False,
                      add_dataloader_idx=False)

    def _generate_structure_viz(self, batch, sc_angs_pred, split):
        sc_angs_pred_rad = inverse_trig_transform(sc_angs_pred, n_angles=6)
        # Select the first protein in the batch to visualize
        j = -1
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
        ptrue.to_pdb(ptrue_pdbfile)

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
        # TODO: Ray tracing occurs despit ray=0
        with Suppressor():
            pymol.cmd.png(both_png_path, width=1000, height=1000, quiet=1, dpi=300, ray=0)
        both_pse_path = os.path.join(self.save_dir, "pdbs", f"{p.id}_both.pse")
        pymol.cmd.save(both_pse_path)
        wandb.save(both_pse_path, base_path=self.save_dir)
        pymol.cmd.delete("all")
        wandb.log({f"structures/{split}/png": wandb.Image(both_png_path)})

    def _compute_openmm_loss(self, batch, sc_angs_pred, loss_dict, split):
        proteins = []
        for (a, s, i, m) in zip(batch.angs, batch.str_seqs, batch.pids, batch.msks):
            p = SCNProtein(angles=a.clone(),
                           sequence=str(s),
                           id=i,
                           mask="".join(["+" if x else "-" for x in m]))
            proteins.append(p)
        loss_total = 0
        sc_angs_pred_rad = inverse_trig_transform(sc_angs_pred, n_angles=6)
        loss_fn = OpenMMEnergyH()
        for p, sca in zip(proteins, sc_angs_pred_rad):
            # print(p.mask)
            # print(p, end=" ")
            # Fill in protein obj with predicted angles instead of true angles
            # print("Set angles")
            p.angles[:, 6:] = sca
            # print("Build coords.")
            p.add_hydrogens(from_angles=True)
            # print("About to apply loss.")
            eloss = loss_fn.apply(p, p.hcoords)
            loss_total += eloss
            # print(eloss.item())
            # del p
        loss = openmm_loss = loss_total / len(proteins)

        loss_dict['loss'] = loss
        loss_dict[self.hparams.loss_name] = loss.detach()

        # Record performance metrics
        self.log(
            split + "/openmm",
            openmm_loss.detach(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        if self.hparams.loss_name == "mse_openmm":
            loss = loss_dict['mse'] * 0.8 + openmm_loss / 1e12 * .2
            self.log(split + "/mse_openmm",
                     loss.detach(),
                     on_step=True,
                     on_epoch=True,
                     prog_bar=True)
        return loss_dict

    def _get_losses(self,
                    batch,
                    sc_angs_true,
                    sc_angs_pred,
                    do_struct=False,
                    split='train'):
        loss_dict = {}
        mse_loss = angle_mse(sc_angs_true, sc_angs_pred)
        if self.hparams.loss_name == "mse":
            loss_dict['loss'] = mse_loss
            loss_dict['mse'] = mse_loss.detach()
        if self.hparams.loss_name == "openmm" or self.hparams.loss_name == "mse_openmm":
            loss_dict = self._compute_openmm_loss(batch, sc_angs_pred, loss_dict, split)
        if do_struct and self.hparams.log_structures:
            self._generate_structure_viz(batch, sc_angs_pred, split)

        loss_dict = self._compute_angle_metrics(sc_angs_true, sc_angs_pred, loss_dict)

        return loss_dict

    def make_example_input_array(self, data_module):
        """Prepare an example input array for batch size determination callback."""
        example_batch = data_module.get_example_input_array()
        non_seq_data = self._prepare_model_input(example_batch)[0]
        self.example_input_array = non_seq_data, example_batch.seqs_int


class LitSCNDataModule(pl.LightningDataModule):
    """A Pytorch Lightning DataModule for SidechainNet data, downstream of scn.load()."""

    def __init__(self, scn_data_dict, batch_size, val_dataloader_target='V50'):
        """Create LitSCFNDataModule from preloaded data.

        Args:
            scn_data_dict (dict): Dictionary mapping train/valid/test splits to their
                respective DataLoaders. In practice, this input is a result of calling
                scn.load(with_pytorch='dataloaders').
            val_dataloader_target (str, optional): The name of the valid dataloader to use
                as a target metric during training. Defaults to 'V50'.
        """
        super().__init__()
        self.scn_data_dict = scn_data_dict
        self.val_dataloader_idx_to_name = {}
        self.val_dataloader_target = val_dataloader_target
        self.batch_size = batch_size

    def get_train_angle_means(self, start_angle=0, end_angle=-1):
        """Return a torch tensor describing the average angle vector of the training set.

        Args:
            start_angle (int, optional): Start position in the per-residue angle array.
                Defaults to 0.
            end_angle (int, optional): End position in the per-residue angle array.
                Defaults to -1.
        """
        means = self.scn_data_dict['train'].dataset.angle_means[start_angle:end_angle]
        n_angles = means.shape[-1]
        means = torch.tensor(means).view(1, 1, len(means))
        means = scn.structure.trig_transform(means).view(1, 1, n_angles * 2).squeeze()
        return means

    def train_dataloader(self):
        """Return the SCN train set dataloader & reset descending order & batch_size."""
        self.scn_data_dict['train'].batch_sampler.turn_off_descending()
        self.scn_data_dict['train'].batch_sampler.batch_size = self.batch_size
        return self.scn_data_dict['train']

    def val_dataloader(self):
        """Return list of SCN validation set dataloaders."""
        vkeys = []
        i = 0
        for key in self.scn_data_dict.keys():
            if "valid" in key:
                renamed_key = key.replace('valid-', 'V')
                vkeys.append(key)
                self.val_dataloader_idx_to_name[i] = renamed_key
                i += 1

        # Set target validation set
        if self.val_dataloader_target not in vkeys:
            self.val_dataloader_target = vkeys[0]

        return [self.scn_data_dict[split_name] for split_name in vkeys]

    def test_dataloader(self):
        """Return SCN test set dataloader."""
        return self.scn_data_dict['test']

    def get_example_input_array(self):
        """Return a sample training batch."""
        batch = next(iter(self.train_dataloader()))
        return batch

    def set_train_dataloader_descending(self):
        """Turn on descending mode for train dataloader."""
        self.scn_data_dict['train'].batch_sampler.make_descending()


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
