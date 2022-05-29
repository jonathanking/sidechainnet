from typing import Dict

import numpy as np
import pytorch_lightning as pl
import radam
import sidechainnet as scn
import torch
from sidechainnet.examples.lightning.AnglePredictionHelper import \
    AnglePredictionHelper
from sidechainnet.examples.lightning.LoggingHelper import LoggingHelper
from sidechainnet.examples.optim import NoamOpt
from sidechainnet.examples.transformer import PositionalEncoding
from sidechainnet.utils.download import MAX_SEQ_LEN
from sidechainnet.utils.sequence import VOCAB


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
        self.automatic_optimization = True
        self.save_hyperparameters()

        # Initialize model architecture
        self._init_layers()

        # Initialize model parameters
        self._init_parameters()
        if angle_means is not None:
            self._init_angle_mean_projection()

        self.log_helper = LoggingHelper(self)

    def _init_layers(self):
        """Initialize the layers for this model's architecture."""
        # Initialize layers
        if self.hparams.embed_sequence:
            self.input_embedding = torch.nn.Embedding(len(VOCAB),
                                                      self.hparams.d_seq_embedding,
                                                      padding_idx=VOCAB.pad_id)
        self.ff1 = torch.nn.Linear(
            self.hparams.d_nonseq_data + self.hparams.d_seq_embedding, self.hparams.d_in)
        self.positional_encoding = PositionalEncoding(d_model=self.hparams.d_in,
                                                      max_len=MAX_SEQ_LEN)
        self.encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.hparams.d_in,
            nhead=self.hparams.n_heads,
            dim_feedforward=self.hparams.d_feedforward,
            dropout=self.hparams.dropout,
            activation=self.hparams.transformer_activation,
            batch_first=True,
            norm_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(
            self.encoder_layer, num_layers=self.hparams.n_layers)
        self.ff2 = torch.nn.Linear(self.hparams.d_in, self.hparams.d_out)
        self.output_activation = torch.nn.Tanh()

    def _init_parameters(self):
        """Initialize layer parameters with Xavier Uniform method."""
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
        x = self.positional_encoding(x)
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
        if self.hparams.opt_name == "adam":
            opt = torch.optim.Adam(filter(lambda x: x.requires_grad, self.parameters()),
                                   lr=lr,
                                   betas=betas,
                                   eps=eps,
                                   weight_decay=self.hparams.opt_weight_decay)
        elif self.hparams.opt_name == "radam":
            opt = radam.RAdam(filter(lambda x: x.requires_grad, self.parameters()),
                              lr=lr,
                              betas=betas,
                              eps=eps,
                              weight_decay=self.hparams.opt_weight_decay)
        elif self.hparams.opt_name == "adamw":
            opt = torch.optim.AdamW(filter(lambda x: x.requires_grad, self.parameters()),
                                    lr=lr,
                                    betas=betas,
                                    eps=eps,
                                    weight_decay=self.hparams.opt_weight_decay)
        elif self.hparams.opt_name == "sgd":
            opt = torch.optim.SGD(filter(lambda x: x.requires_grad, self.parameters()),
                                  lr=lr,
                                  weight_decay=self.hparams.opt_weight_decay,
                                  momentum=0.9)

        # Prepare scheduler
        if (self.hparams.opt_lr_scheduling == "noam" and
                self.hparams.opt_name in ['adam', 'adamw']):
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
        else:
            sch = None

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
        batch.set_device(self.device)
        # True values still have nans, replace with 0s so they can go into the network
        # Also select out backbone and sidechain angles
        bb_angs = torch.nan_to_num(batch.angles[:, :, :6], nan=0.0)
        sc_angs_true_untransformed = batch.angles[:, :, 6:]

        # Since *batches* are padded with 0s, we replace with nan for convenient loss fns
        sc_angs_true_untransformed[sc_angs_true_untransformed.eq(0).all(dim=-1)] = np.nan

        # Result after transform (6 angles, sin/cos): (B x L x 12)
        sc_angs_true = scn.structure.trig_transform(sc_angs_true_untransformed).reshape(
            sc_angs_true_untransformed.shape[0], sc_angs_true_untransformed.shape[1], 12)

        # Stack model inputs into a single tensor  # TODO normalize the input, esp angs
        model_in = torch.cat([bb_angs, batch.secondary, batch.evolutionary], dim=-1)

        return model_in, sc_angs_true

    def training_step(self, batch, batch_idx):
        """Perform a single step of training (model in, model out, log loss).

        Args:
            batch (List): List of Protein objects.
            batch_idx (int): Integer index of the batch.
        """
        model_in, sc_angs_true = self._prepare_model_input(batch)

        # Predict sidechain angles given input and sequence
        sc_angs_pred = self(model_in, batch.seqs_int)  # ( B x L x 12)

        # Create a helper obj to analyze preds; remember `batch` is metadata, not tensors
        pred_helper = AnglePredictionHelper(batch, sc_angs_true, sc_angs_pred)

        # Compute loss and step
        loss_dict = self._get_losses(pred_helper,
                                     do_struct=batch_idx == 0,
                                     split='train')
        # Log metrics
        self.log_helper.log_training_step(loss_dict, batch)

        if loss_dict['loss'] is None:
            return None

        if not self.automatic_optimization:
            opt = self.optimizers()
            opt.zero_grad()
            self.manual_backward(loss_dict['loss'])
            torch.nn.utils.clip_grad_value_(self.parameters(), 0.5)
            opt.step()

        return_vals = {
            "model_in": model_in,
            "pred_helper": pred_helper
        }
        return_vals.update(loss_dict)

        return return_vals

    def validation_step(self,
                        batch,
                        batch_idx,
                        dataloader_idx=0) -> Dict[str, torch.Tensor]:
        """Single validation step with multiple possible DataLoaders."""
        model_in, sc_angs_true = self._prepare_model_input(batch)

        # Predict sidechain angles given input and sequence
        sc_angs_pred = self(model_in, batch.seqs_int)  # ( B x L x 12)

        # Create a helper obj to analyze preds; remember `batch` is metadata, not tensors
        pred_helper = AnglePredictionHelper(batch, sc_angs_true, sc_angs_pred)

        # Compute loss
        loss_dict = self._get_losses(pred_helper,
                                     do_struct=batch_idx == 0 and dataloader_idx == 0,
                                     split='valid')
        self.log_helper.log_validation_step(loss_dict, dataloader_idx)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        """Single test step with multiple possible DataLoaders. Same as validation."""
        model_in, sc_angs_true = self._prepare_model_input(batch)

        # Predict sidechain angles given input and sequence
        sc_angs_pred = self(model_in, batch.seqs_int)  # ( B x L x 12)

        # Create a helper obj to analyze preds; remember `batch` is metadata, not tensors
        pred_helper = AnglePredictionHelper(batch, sc_angs_true, sc_angs_pred)

        # Compute loss
        loss_dict = self._get_losses(pred_helper,
                                     do_struct=batch_idx == 0,
                                     split='test')
        self.log_helper.log_test_step(loss_dict)

    def _get_losses(self,
                    pred_helper,
                    do_struct=False,
                    split='train'):

        loss_dict = {}
        mse_loss = pred_helper.angle_mse()
        loss_dict['mse'] = mse_loss.detach()

        # Basic structure-based losses/metrics
        loss_dict['rmsd'] = pred_helper.rmsd()
        loss_dict['drmsd'] = pred_helper.drmsd()
        loss_dict['lndrmsd'] = pred_helper.lndrmsd()
        loss_dict['gdc_all'] = pred_helper.gdc_all()

        # Loss values (mse, OpenMM)
        if self.hparams.loss_name == "mse":
            loss_dict['loss'] = mse_loss

        if self.hparams.loss_name == "openmm":
            openmm_loss = pred_helper.openmm_loss()
            loss_dict['openmm'] = openmm_loss.detach()
            loss_dict['loss'] = openmm_loss

        if self.hparams.loss_name == "mse_openmm":
            if self.global_step < self.hparams.opt_begin_mse_openmm_step:  # or self.global_step % 2 == 0:
                loss_dict['loss'] = mse_loss
            else:
                if self.global_step == self.hparams.opt_begin_mse_openmm_step:
                    print("Starting to train with OpenMM/MSE Combination loss at step",
                          self.global_step)
                openmm_loss = pred_helper.openmm_loss()
                if openmm_loss is None:
                    loss_dict['openmm'] = None
                    loss_dict['loss'] = None
                else:
                    loss_dict['openmm'] = openmm_loss.detach()
                    a = self.hparams.loss_combination_weight
                    b = 1 - a
                    loss = mse_loss * a + openmm_loss / 1e12 * b
                    loss_dict['loss'] = loss
                    # Scale the value of the loss significantly
                    if torch.log10(loss) - 1 > 0:
                        loss_dict['loss'] = loss / 10**(
                            torch.floor(torch.log10(loss) - 1))
                    loss_dict['mse_openmm'] = loss_dict['loss'].detach()

        loss_dict.update(pred_helper.angle_metrics_dict())

        # Generate structures only after we no longer need the objects intact
        # TODO Remove out of date structure viz arguments etc
        # if do_struct and self.hparams.log_structures:
        #     self._generate_structure_viz(batch, sc_angs_pred, split)

        return loss_dict

    def make_example_input_array(self, data_module):
        """Prepare an example input array for batch size determination callback."""
        example_batch = data_module.get_example_input_array()
        non_seq_data = self._prepare_model_input(example_batch)[0]
        self.example_input_array = non_seq_data, example_batch.seqs_int
