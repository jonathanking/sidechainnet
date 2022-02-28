import sys
from typing import Dict
import pymol
import copy
import os
import numpy as np
import torch
from sidechainnet.dataloaders.SCNProtein import SCNProtein
from sidechainnet.examples.losses import angle_mse
from sidechainnet.structure.structure import inverse_trig_transform
from sidechainnet.utils.openmm import OpenMMEnergyH
from sidechainnet.examples.optim import ScheduledOptim

import pytorch_lightning as pl

from sidechainnet.utils.sequence import VOCAB
import sidechainnet as scn

import wandb

# pymol.finish_launching(['pymol', '-qc'])

# TODO move model-specific argparse arges to lightning module
# https://pytorch-lightning.readthedocs.io/en/stable/common/hyperparameters.html


class LitSidechainTransformer(pl.LightningModule):
    """PyTorch Lightning module for SidechainTransformer."""

    def __init__(
            self,
            d_seq_embedding=20,
            d_nonseq_data=35,  # 5 bb, 21 PSSM, 8 ss
            d_in=256,
            d_out=6,
            d_feedforward=1024,
            n_heads=8,
            n_layers=1,
            dropout=0,
            activation='relu',
            angle_means=None,
            embed_sequence=True,
            loss_name=None,
            opt_name='adam',
            opt_lr=1e-2,
            opt_lr_scheduling='plateau',
            opt_lr_scheduling_metric='val_loss',
            opt_patience=5,
            opt_early_stopping_threshold=0.01,
            opt_weight_decay=1e-5,
            opt_n_warmup_steps=5_000,
            dataloader_name_mapping=None):
        """Create a LitSidechainTransformer module."""
        super().__init__()
        assert loss_name is not None, "Please provide a loss for optimization."
        self.embed_sequence = embed_sequence
        if self.embed_sequence:
            self.input_embedding = torch.nn.Embedding(len(VOCAB),
                                                      d_seq_embedding,
                                                      padding_idx=VOCAB.pad_id)
        self.angle_means = angle_means
        if dataloader_name_mapping is None:
            self.dataloader_name_mapping = {0: ""}
        else:
            self.dataloader_name_mapping = dataloader_name_mapping
        # Initialize layers
        self._d_model = d_nonseq_data + d_seq_embedding
        while d_in % n_heads != 0:
            n_heads -= 1
        self.n_heads = n_heads
        self.ff1 = torch.nn.Linear(d_nonseq_data + d_seq_embedding, d_in)
        self.encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_in,
            nhead=self.n_heads,
            dim_feedforward=d_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(self.encoder_layer,
                                                               num_layers=n_layers)
        self.ff2 = torch.nn.Linear(d_in, d_out)
        self.output_activation = torch.nn.Tanh()

        self.loss_name = loss_name

        if self.angle_means is not None:
            self._init_parameters()

        # Optimizer options
        self.opt_name = opt_name
        self.opt_lr = opt_lr
        self.opt_lr_scheduling = opt_lr_scheduling
        self.opt_lr_scheduling_metric = opt_lr_scheduling_metric
        self.opt_patience = opt_patience
        self.opt_early_stopping_threshold = opt_early_stopping_threshold
        self.opt_weight_decay = opt_weight_decay
        self.opt_n_warmup_steps = opt_n_warmup_steps

        self.save_dir = ""

        self.save_hyperparameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
        angle_means = np.arctanh(self.angle_means)
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
        if self.embed_sequence:
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
        if self.opt_name == "adam":
            opt = torch.optim.Adam(filter(lambda x: x.requires_grad, self.parameters()),
                                   lr=self.opt_lr,
                                   weight_decay=self.opt_weight_decay)
        elif self.opt_name == "adamw":
            opt = torch.optim.AdamW(filter(lambda x: x.requires_grad, self.parameters()),
                                    lr=self.opt_lr,
                                    weight_decay=self.opt_weight_decay)
        elif self.opt_name == "sgd":
            opt = torch.optim.SGD(filter(lambda x: x.requires_grad, self.parameters()),
                                  lr=self.opt_lr,
                                  weight_decay=self.opt_weight_decay)

        # Prepare scheduler
        if self.opt_lr_scheduling == "noam":
            opt = ScheduledOptim(opt, self.d_in, self.n_warmup_steps)
            sch = None
            # TODO: Use better warm up scheduler
        else:
            sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                patience=self.opt_patience,
                verbose=True,
                threshold=self.opt_early_stopping_threshold)

        d = {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sch,
                "interval": "epoch",
                "frequency": 1,
                "monitor": self.opt_lr_scheduling_metric,
                "strict": True,
                "name": None,
            }
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
        # print("prepare", model_in.device, sc_angs_true.device)

        return model_in.cuda(), sc_angs_true.cuda()

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
            'train/rmse',
            torch.sqrt(loss_dict['mse']),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

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
                                     split='valid',
                                     dataloader_idx=dataloader_idx)

        name = f"valid/{self.dataloader_name_mapping[dataloader_idx]}_rmse"
        self.log(name,
                 torch.sqrt(loss_dict['mse']),
                 on_step=False,
                 on_epoch=True,
                 prog_bar=name == self.opt_lr_scheduling_metric,
                 add_dataloader_idx=False)

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

        self.log("test/rmse", torch.sqrt(loss_dict['mse']), on_step=True, on_epoch=True)

    def _get_losses(self,
                    batch,
                    sc_angs_true,
                    sc_angs_pred,
                    do_struct=False,
                    split='train',
                    dataloader_idx=None):
        loss_dict = {}
        mse_loss = angle_mse(sc_angs_true, sc_angs_pred)
        if self.loss_name == "mse":
            # loss = mse_loss
            loss_dict['loss'] = mse_loss
            loss_dict[self.loss_name] = mse_loss.detach()
        if self.loss_name == "openmm" or self.loss_name == "mse_openmm":

            proteins = []
            for (a, s, i, m) in zip(batch.angs, batch.str_seqs, batch.pids, batch.msks):
                p = SCNProtein(angles=a.clone(),
                               sequence=str(s),
                               id=i,
                               mask="".join(["+" if x else "-" for x in m]))
                proteins.append(p)
            loss_total = 0
            sc_angs_pred_untransformed = inverse_trig_transform(sc_angs_pred, n_angles=6)
            loss_fn = OpenMMEnergyH()
            for p, sca in zip(proteins, sc_angs_pred_untransformed):
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
            loss_dict[self.loss_name] = loss.detach()

            # Record performance metrics
            self.log(
                split + "/openmm",
                openmm_loss.detach(),
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )
            if self.loss_name == "mse_openmm":
                loss = mse_loss * 0.8 + openmm_loss / 1e12 * .2
                self.log(split + "/mse_openmm",
                         loss.detach(),
                         on_step=True,
                         on_epoch=True,
                         prog_bar=True)

        if do_struct:
            sc_angs_pred_untransformed = inverse_trig_transform(sc_angs_pred, n_angles=6)
            p = copy.deepcopy(batch[0])
            p.angles[:, 6:] = sc_angs_pred_untransformed[0,
                                                         0:len(p)].detach().cpu().numpy()
            p.numpy()
            p.build_coords_from_angles(
            )  # Make sure the coordinates saved to PDB are updated
            pdbfile = os.path.join(self.save_dir, "pdbs", f"{p.id}_pred.pdb")
            p.to_pdb(pdbfile)
            wandb.log({"structures/molecule": wandb.Molecule(pdbfile)})

            # Now to generate the files for the true structure
            ptrue = batch[0]
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
                pymol.cmd.png(both_png_path,
                              width=1000,
                              height=1000,
                              quiet=1,
                              dpi=300,
                              ray=0)
            both_pse_path = os.path.join(self.save_dir, "pdbs", f"{p.id}_both.pse")
            pymol.cmd.save(both_pse_path)
            wandb.save(both_pse_path, base_path=self.save_dir)
            pymol.cmd.delete("all")
            self.logger.log_image(key="structures/png",
                                  images=[wandb.Image(both_png_path)])

        return loss_dict


class LitSCNDataModule(pl.LightningDataModule):

    def __init__(self, scn_data_dict, val_dataloader_target='valid-10'):
        super().__init__()
        self.scn_data_dict = scn_data_dict
        self.val_dataloader_idx_to_name = {}
        self.val_dataloader_target = val_dataloader_target

    def train_dataloader(self):
        return self.scn_data_dict['train']

    def val_dataloader(self):
        vkeys = []
        i = 0
        for key in self.scn_data_dict.keys():
            if "valid" in key:
                vkeys.append(key)
                self.val_dataloader_idx_to_name[i] = key
                i += 1

        # Set target validation set
        if self.val_dataloader_target not in vkeys:
            self.val_dataloader_target = vkeys[0]

        return [self.scn_data_dict[split_name] for split_name in vkeys]

    def test_dataloader(self):
        return self.scn_data_dict['test']

    def get_example_input_array(self):
        batch = next(iter(self.train_dataloader()))
        return batch


class Suppressor(object):

    def __enter__(self):
        devnull = open('/dev/null', 'w')
        self.oldstdout_fno = os.dup(sys.stdout.fileno())
        os.dup2(devnull.fileno(), 1)

    def __exit__(self, type, value, traceback):
        os.dup2(self.oldstdout_fno, 1)

    def write(self, x):
        pass
