"""Example models for training protein sequence to angle coordinates tasks."""

import numpy as np
import torch
from sidechainnet.structure.build_info import NUM_ANGLES


class BaseProteinAngleRNN(torch.nn.Module):
    """A simple RNN that consumes protein sequences and produces angles."""

    def __init__(self,
                 size,
                 n_layers=1,
                 d_in=20,
                 d_out=NUM_ANGLES,
                 bidirectional=True,
                 device=torch.device('cpu')):
        super(BaseProteinAngleRNN, self).__init__()
        self.size = size
        self.n_layers = n_layers
        self.lstm = torch.nn.LSTM(d_in,
                                  size,
                                  n_layers,
                                  bidirectional=bidirectional,
                                  batch_first=True)
        self.n_direction = 2 if bidirectional else 1
        self.hidden2out = torch.nn.Linear(self.n_direction * size, d_out)
        self.output_activation = torch.nn.Tanh()
        self.device = device
        self.d_out = d_out

    def init_hidden(self, batch_size):
        """Initialize the hidden state vectors at the start of a batch iteration."""
        h, c = (torch.zeros(self.n_layers * self.n_direction, batch_size,
                            self.size).to(self.device),
                torch.zeros(self.n_layers * self.n_direction, batch_size,
                            self.size).to(self.device))
        return h, c

    def forward(self, *args, **kwargs):
        """Run one forward step of the model."""
        raise NotImplementedError


class IntegerSequenceProteinRNN(BaseProteinAngleRNN):
    """A protein sequence-to-angle model that consumes integer-coded sequences."""

    def __init__(self,
                 size,
                 n_layers=1,
                 d_in=20,
                 d_out=NUM_ANGLES,
                 bidirectional=True,
                 device=torch.device('cpu')):
        super(IntegerSequenceProteinRNN, self).__init__(size, n_layers, d_in, d_out,
                                                        bidirectional, device)

        self.input_embedding = torch.nn.Embedding(21, 20, padding_idx=20)

    def forward(self, sequence):
        """Run one forward step of the model."""
        # First, we compute the length of each sequence to use pack_padded_sequence
        lengths = sequence.shape[-1] - (sequence == 20).sum(axis=1)
        h, c = self.init_hidden(len(lengths))
        # Our inputs are sequences of integers, allowing us to use torch.nn.Embeddings
        sequence = self.input_embedding(sequence)
        sequence = torch.nn.utils.rnn.pack_padded_sequence(sequence,
                                                           lengths,
                                                           batch_first=True,
                                                           enforce_sorted=False)
        output, (h, c) = self.lstm(sequence, (h, c))
        output, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(output,
                                                                        batch_first=True)
        output = self.hidden2out(output)
        # We push the output through a tanh layer and multiply by pi to ensure
        # values are within [-pi, pi].
        output = self.output_activation(output) * np.pi
        output = output.view(output.shape[0], output.shape[1], self.d_out)
        return output


class PSSMProteinRNN(BaseProteinAngleRNN):
    """A protein sequence-to-angle model that consumes 1-hot sequences and PSSMs."""

    def __init__(self,
                 size,
                 n_layers=1,
                 d_in=41,
                 d_out=NUM_ANGLES,
                 bidirectional=True,
                 device=torch.device('cpu')):
        """Create a PSSMSequenceProteinRNN model with input dimensionality 41."""
        super(PSSMProteinRNN, self).__init__(size, n_layers, d_in, d_out,
                                             bidirectional, device)

    def forward(self, sequence):
        """Run one forward step of the model."""
        # First, we compute the length of each sequence to use pack_padded_sequence
        lengths = sequence.shape[1] - (sequence == 0).all(axis=2).sum(axis=1)
        h, c = self.init_hidden(len(lengths))
        sequence = torch.nn.utils.rnn.pack_padded_sequence(sequence,
                                                           lengths,
                                                           batch_first=True,
                                                           enforce_sorted=False)
        output, (h, c) = self.lstm(sequence, (h, c))
        output, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(output,
                                                                        batch_first=True)
        output = self.hidden2out(output)
        # We push the output through a tanh layer and multiply by pi to ensure
        # values are within [-pi, pi].
        output = self.output_activation(output) * np.pi
        output = output.view(output.shape[0], output.shape[1], self.d_out)
        return output
