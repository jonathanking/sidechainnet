import numpy as np
import torch


class SidechainTransformer(torch.nn.Module):
    """A Transformer designed to tale protein sequence data and emit sidechain angles."""

    def __init__(self,
                 d_in=6,
                 d_out=6,
                 nhead=8,
                 nlayers=1,
                 dim_feedforward=1024,
                 dropout=0,
                 activation='relu',
                 batch_first=True,
                 device='cpu',
                 angle_means=None):
        super(SidechainTransformer, self).__init__()
        self.encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_in,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=batch_first,
            device=device)
        self.transformer_encoder = torch.nn.TransformerEncoder(self.encoder_layer,
                                                               num_layers=nlayers)
        self.ff = torch.nn.Linear(d_in, d_out)
        self.output_activation = torch.nn.Tanh()
        other_inputs = 35  # 6 bb angles, 21 PSSMs, 8 secondary str
        self.input_embedding = torch.nn.Embedding(21, d_in - other_inputs, padding_idx=20)
        self.angle_means = angle_means

        if self.angle_means is not None:
            self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
        self.ff.bias = torch.nn.Parameter(
            torch.tensor(np.arctanh(self.angle_means), dtype=torch.float32))
        torch.nn.init.zeros_(self.ff.weight)

    def forward(self, x, seq):
        """Run one forward step of the model."""
        embedded_seq = self.input_embedding(seq)
        x = torch.cat([x, embedded_seq], dim=-1)
        x = self.transformer_encoder(x)
        x = self.ff(x)
        x = self.output_activation(x)
        return x
