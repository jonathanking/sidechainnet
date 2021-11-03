import numpy as np
import torch


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
            batch_first=True,
            device='cpu',
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
            activation=activation)  # ,
        # batch_first=batch_first,
        # device=device)
        self.transformer_encoder = torch.nn.TransformerEncoder(self.encoder_layer,
                                                               num_layers=n_layers)
        self.ff2 = torch.nn.Linear(d_in, d_out)
        self.output_activation = torch.nn.Tanh()
        self.embed_sequence = embed_sequence
        if self.embed_sequence:
            self.input_embedding = torch.nn.Embedding(21, d_seq_embedding,
                                                      padding_idx=20)
        self.angle_means = angle_means

        if self.angle_means is not None:
            self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
        angle_means = np.arctanh(self.angle_means)
        angle_means[~torch.isfinite(angle_means)] = 0
        self.ff2.bias = torch.nn.Parameter(angle_means)
        torch.nn.init.zeros_(self.ff2.weight)

    def forward(self, x, seq):
        """Run one forward step of the model."""
        if self.embed_sequence:
            seq = self.input_embedding(seq)
        # print(torch.isnan(x).any(), "1")
        x = torch.cat([x, seq], dim=-1)
        # print(torch.isnan(x).any(), "2")
        x = self.ff1(x)
        # print(torch.isnan(x).any(), "3")
        x = self.transformer_encoder(x)
        # print(torch.isnan(x).any(), "4")
        x = self.ff2(x)
        # print(torch.isnan(x).any(), "5")
        x = self.output_activation(x)
        # print(torch.isnan(x).any(), "6")
        return x
