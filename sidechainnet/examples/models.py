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
                 n_angles=NUM_ANGLES,
                 bidirectional=True,
                 sincos_output=True,
                 device=torch.device('cpu')):
        super(BaseProteinAngleRNN, self).__init__()
        self.size = size
        self.n_layers = n_layers
        self.sincos_output = sincos_output
        self.d_out = n_angles * 2 if sincos_output else n_angles
        self.lstm = torch.nn.LSTM(d_in,
                                  size,
                                  n_layers,
                                  bidirectional=bidirectional,
                                  batch_first=True)
        self.n_direction = 2 if bidirectional else 1
        self.hidden2out = torch.nn.Linear(self.n_direction * size, self.d_out)
        self.output_activation = torch.nn.Tanh()
        self.device_ = device

    def init_hidden(self, batch_size):
        """Initialize the hidden state vectors at the start of a batch iteration."""
        h, c = (torch.zeros(self.n_layers * self.n_direction, batch_size,
                            self.size).to(self.device_),
                torch.zeros(self.n_layers * self.n_direction, batch_size,
                            self.size).to(self.device_))
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
                 n_angles=NUM_ANGLES,
                 bidirectional=True,
                 device=torch.device('cpu'),
                 sincos_output=True):
        super(IntegerSequenceProteinRNN, self).__init__(size=size,
                                                        n_layers=n_layers,
                                                        d_in=d_in,
                                                        n_angles=n_angles,
                                                        bidirectional=bidirectional,
                                                        device=device,
                                                        sincos_output=sincos_output)

        self.input_embedding = torch.nn.Embedding(21, 20, padding_idx=20)

    def forward(self, sequence):
        """Run one forward step of the model."""
        # First, we compute the length of each sequence to use pack_padded_sequence
        lengths = sequence.shape[-1] - (sequence == 20).sum(axis=1)
        h, c = self.init_hidden(len(lengths))
        # Our inputs are sequences of integers, allowing us to use torch.nn.Embeddings
        sequence = self.input_embedding(sequence)
        sequence = torch.nn.utils.rnn.pack_padded_sequence(sequence,
                                                           lengths.cpu(),
                                                           batch_first=True,
                                                           enforce_sorted=False)
        output, (h, c) = self.lstm(sequence, (h, c))
        output, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(output,
                                                                        batch_first=True)
        output = self.hidden2out(output)
        if self.sincos_output:
            output = self.output_activation(output)
            output = output.view(output.shape[0], output.shape[1], int(self.d_out / 2), 2)
        else:
            # We push the output through a tanh layer and multiply by pi to ensure
            # values are within [-pi, pi] for predicting raw angles.
            output = self.output_activation(output) * np.pi
            output = output.view(output.shape[0], output.shape[1], self.d_out)
        return output


class PSSMProteinRNN(BaseProteinAngleRNN):
    """A protein structure model consuming 1-hot sequences, 2-ary structures, & PSSMs."""

    def __init__(self,
                 size,
                 n_layers=1,
                 d_in=49,
                 n_angles=NUM_ANGLES,
                 bidirectional=True,
                 device=torch.device('cpu'),
                 sincos_output=True):
        """Create a PSSMProteinRNN model with input dimensionality 41."""
        super(PSSMProteinRNN, self).__init__(size=size,
                                             n_layers=n_layers,
                                             d_in=d_in,
                                             n_angles=n_angles,
                                             bidirectional=bidirectional,
                                             device=device,
                                             sincos_output=sincos_output)

    def forward(self, sequence):
        """Run one forward step of the model."""
        # First, we compute the length of each sequence to use pack_padded_sequence
        lengths = sequence.shape[1] - (sequence == 0).all(axis=2).sum(axis=1)
        h, c = self.init_hidden(len(lengths))
        sequence = torch.nn.utils.rnn.pack_padded_sequence(sequence,
                                                           lengths.cpu(),
                                                           batch_first=True,
                                                           enforce_sorted=False)
        output, (h, c) = self.lstm(sequence, (h, c))
        output, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(output,
                                                                        batch_first=True)
        output = self.hidden2out(output)
        if self.sincos_output:
            output = self.output_activation(output)
            output = output.view(output.shape[0], output.shape[1], int(self.d_out / 2), 2)
        else:
            # We push the output through a tanh layer and multiply by pi to ensure
            # values are within [-pi, pi] for predicting raw angles.
            output = self.output_activation(output) * np.pi
            output = output.view(output.shape[0], output.shape[1], self.d_out)
        return output
