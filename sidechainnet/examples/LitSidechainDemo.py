from typing import Dict

import numpy as np
import pytorch_lightning as pl
import torch
from sidechainnet.examples.losses import angle_diff, angle_mse
from sidechainnet.examples.optim import NoamOpt
from sidechainnet.structure.build_info import ANGLE_IDX_TO_NAME_MAP
from sidechainnet.structure.structure import inverse_trig_transform
from sidechainnet.utils.sequence import VOCAB
import sidechainnet as scn


class LitSidechainTransformerBaseModule(pl.LightningModule):
    """PyTorch Lightning module for SidechainTransformer."""

    def __init__(self):
        """Create a LitSidechainTransformer module."""
        super().__init__()

    def _get_seq_pad_mask(self, seq):
        # Seq is Batch x L
        assert len(seq.shape) == 2
        return seq == VOCAB.pad_id

    def forward(self, x, seq):
        """Run one forward step of the model."""
        raise NotImplementedError

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
        # Setup default optimizer construction values
        if self.hparams.opt_lr_scheduling == "noam":
            lr = 0
            betas = (0.9, 0.98)
            eps = 1e-9
        else:
            lr = self.hparams.opt_lr
            betas = (0.9, 0.999)
            eps = 1e-8

        # Prepare optimizer
        opt = torch.optim.Adam(filter(lambda x: x.requires_grad, self.parameters()),
                               lr=lr,
                               betas=betas,
                               eps=eps,
                               weight_decay=self.hparams.opt_weight_decay)

        # Prepare scheduler
        if (self.hparams.opt_lr_scheduling == "noam"):
            opt = NoamOpt(model_size=self.hparams.d_in,
                          warmup=self.hparams.opt_n_warmup_steps,
                          optimizer=opt,
                          factor=self.hparams.opt_noam_lr_factor)
            sch = None
        elif self.hparams.opt_lr_scheduling == 'plateau':
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
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        """Perform a single step of training (model in, model out, log loss).

        Args:
            batch (List): List of Protein objects.
            batch_idx (int): Integer index of the batch.
        """
        raise NotImplementedError

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

        # Compute Accuracy per Angle (APA) which evaluates each angle indp.
        ang_is_predicted = ~torch.isnan(correct_angle_preds)
        ang_is_correct = torch.logical_or(correct_angle_preds, torch.isnan(abs_error))
        num_correct_angles = torch.logical_and(ang_is_predicted, ang_is_correct).sum()
        apa = num_correct_angles / ang_is_predicted.sum()
        loss_dict['angle_metrics']['apa'] = apa

        return loss_dict

    def _log_angle_metrics(self, loss_dict, split, dataloader_name=""):
        dl_name_str = dataloader_name + "_" if dataloader_name else ""

        angle_metrics_dict = {
            f"metrics/{split}/{dl_name_str}{k}": loss_dict['angle_metrics'][k]
            for k in loss_dict['angle_metrics'].keys()
        }
        self.log_dict(angle_metrics_dict,
                      on_epoch=True,
                      on_step=split == 'train',
                      add_dataloader_idx=False)

    def _get_losses(self,
                    batch,
                    sc_angs_true,
                    sc_angs_pred,
                    do_struct=False,
                    split='train'):

        loss_dict = {}
        mse_loss = angle_mse(sc_angs_true, sc_angs_pred)
        loss_dict['mse'] = mse_loss.detach()
        loss_dict['loss'] = mse_loss

        loss_dict = self._compute_angle_metrics(sc_angs_true, sc_angs_pred, loss_dict)

        return loss_dict


class LitSCNDataModule(pl.LightningDataModule):
    """A Pytorch Lightning DataModule for SidechainNet data, downstream of scn.load()."""

    def __init__(self, scn_data_dict, batch_size, val_dataloader_target='V50'):
        """Create LitSCNDataModule from preloaded data.

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
