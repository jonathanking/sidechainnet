import numpy as np
import torch

# import pytorch_lightning as pl

from sidechainnet.utils.sequence import VOCAB


class SidechainTransformer(torch.nn.Module):
    """A Transformer designed to talk protein sequence data and emit sidechain angles."""

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
            embed_sequence=True):
        super(SidechainTransformer, self).__init__()

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
        self.embed_sequence = embed_sequence
        if self.embed_sequence:
            self.input_embedding = torch.nn.Embedding(len(VOCAB),
                                                      d_seq_embedding,
                                                      padding_idx=VOCAB.pad_id)
        self.angle_means = angle_means

        if self.angle_means is not None:
            self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
        angle_means = np.arctanh(self.angle_means)
        self.ff2.bias = torch.nn.Parameter(angle_means)
        torch.nn.init.zeros_(self.ff2.weight)
        self.ff2.bias.requires_grad_ = False

    def get_seq_pad_mask(self, seq):
        # Seq is Batch x L
        assert len(seq.shape) == 2
        return seq == VOCAB.pad_id

    def forward(self, x, seq):
        """Run one forward step of the model."""
        padding_mask = self.get_seq_pad_mask(seq)
        if self.embed_sequence:
            seq = self.input_embedding(seq)
        x = torch.cat([x, seq], dim=-1)
        x = self.ff1(x)
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        x = self.ff2(x)
        x = self.output_activation(x)
        return x


# class LitSidechainTransformer(pl.LightningModule):
# class LitSidechainTransformer(object):

#     def __init__(
#             self,
#             d_seq_embedding=20,
#             d_nonseq_data=35,  # 5 bb, 21 PSSM, 8 ss
#             d_in=256,
#             d_out=6,
#             d_feedforward=1024,
#             n_heads=8,
#             n_layers=1,
#             dropout=0,
#             activation='relu',
#             angle_means=None,
#             embed_sequence=True):
#         super().__init__()

#         self._d_model = d_nonseq_data + d_seq_embedding
#         while d_in % n_heads != 0:
#             n_heads -= 1
#         self.n_heads = n_heads
#         self.ff1 = torch.nn.Linear(d_nonseq_data + d_seq_embedding, d_in)
#         self.encoder_layer = torch.nn.TransformerEncoderLayer(
#             d_model=d_in,
#             nhead=self.n_heads,
#             dim_feedforward=d_feedforward,
#             dropout=dropout,
#             activation=activation,
#             batch_first=True,
#             norm_first=True)
#         self.transformer_encoder = torch.nn.TransformerEncoder(self.encoder_layer,
#                                                                num_layers=n_layers)
#         self.ff2 = torch.nn.Linear(d_in, d_out)
#         self.output_activation = torch.nn.Tanh()
#         self.embed_sequence = embed_sequence
#         if self.embed_sequence:
#             self.input_embedding = torch.nn.Embedding(21, d_seq_embedding, padding_idx=20)
#         self.angle_means = angle_means

#         if self.angle_means is not None:
#             self._init_parameters()

#     def _init_parameters(self):
#         for p in self.parameters():
#             if p.dim() > 1:
#                 torch.nn.init.xavier_uniform_(p)
#         angle_means = np.arctanh(self.angle_means)
#         self.ff2.bias = torch.nn.Parameter(angle_means)
#         torch.nn.init.zeros_(self.ff2.weight)
#         self.ff2.bias.requires_grad_ = False

#     def forward(self, x, seq):
#         """Run one forward step of the model."""
#         if self.embed_sequence:
#             seq = self.input_embedding(seq)
#         x = torch.cat([x, seq], dim=-1)
#         x = self.ff1(x)
#         x = self.transformer_encoder(x)
#         x = self.ff2(x)
#         x = self.output_activation(x)
#         return x

#     # Lightning Hooks

#     def configure_optimizers(self,
#                              optimizer,
#                              learning_rate,
#                              weight_decay=True,
#                              patience=3,
#                              threshold=0.01,
#                              metric_to_track='val_loss'):
#         """Prepare optimizer and schedulers.

#         Args:
#             optimizer (str): Name of optimizer ('adam', 'sgd')
#             learning_rate (float): Learning rate for optimizer.
#             weight_decay (bool, optional): Use optimizer weight decay. Defaults to True.

#         Returns:
#             dict: Pytorch Lightning dictionary with keys "optimizer" and "lr_scheduler".
#         """
#         # Prepare optimizer
#         wd = 10e-3 if weight_decay else 0
#         if optimizer == "adam":
#             opt = torch.optim.Adam(filter(lambda x: x.requires_grad, self.parameters()),
#                                    lr=learning_rate,
#                                    weight_decay=wd)
#         elif optimizer == "sgd":
#             opt = torch.optim.SGD(filter(lambda x: x.requires_grad, self.parameters()),
#                                   lr=learning_rate,
#                                   weight_decay=wd)

#         # Prepare scheduler
#         sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,
#                                                          patience=patience,
#                                                          verbose=True,
#                                                          threshold=threshold)

#         d = {
#             "optimizer": opt,
#             "lr_scheduler": {
#                 "scheduler": sch,
#                 "interval": "epoch",
#                 "frequency": 1,
#                 "monitor": metric_to_track,
#                 "strict": True,
#                 "name": None,
#             }
#         }

#         return d

#     def training_step(self, batch, batch_idx):
#         """Perform a single step of training (model in, model out, log loss.)

#         Args:
#             batch (List): List of Protein objects.
#             batch_idx (int): Integer index of the batch.
#         """
#         pass