import numpy as np
import torch

from sidechainnet.utils.sequence import VOCAB

# TODO: PyTorch Lightning do not forgets:
# TODO: -clip gradients using the Trainer (clipval1)
# TODO: -use wandb as the registered logger

LOCAL_DIR = ""


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


# class _SCNDataModuleFull(pl.LightningDataModule):

#     def __init__(self, scn_data_dict, batch_size=32, num_workers=1, dynamic_batching=False, optimize_for_cpu_parallelism=False):
#         super().__init__()
#         self.scn_data_dict = scn_data_dict
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.dynamic_batching = dynamic_batching
#         self.dynamic_batch_size = self.batch_size *  self.scn_data_dict['settings']['lengths'].mean()
#         self.optimize_for_cpu_parallelism = optimize_for_cpu_parallelism
#         # TODO: Fix shuffling

#         if self.dynamic_batching:
#             self.dynamic_batch_size = self.batch_size * self.scn_data_dict['settings'][
#                 'lengths'].mean()
#         else:
#             self.dynamic_batch_size = None

#     def _get_collate_fn(self):
#         def collate(protein_objects):
#             return ProteinBatch(protein_objects)

#     def train_dataloader(self):
#         train_dataset = SCNDataset(self.scn_data_dict['train'], split_name='train')
#         sampler = SimilarLengthBatchSampler(
#                 train_dataset,
#                 self.batch_size,
#                 dynamic_batch=self.dynamic_batch_size,
#                 optimize_batch_for_cpus=self.optimize_for_cpu_parallelism,
#                 shuffle=False)
#         train_dl = torch.utils.data.DataLoader(
#             train_dataset,
#             num_workers=self.num_workers,
#             collate_fn=self._get_collate_fn(),
#             batch_sampler=sampler)
#         return train_dl

#     def val_dataloader(self):
#         vkeys = []
#         for key in self.scn_dataset.keys():
#             if "valid" in key:
#                 vkeys.append(key)
#         return [self.scn_dataset[split_name] for split_name in vkeys]

#     def test_dataloader(self):
#         return self.scn_dataset['test']
