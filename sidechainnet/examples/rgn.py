import numpy as np
import torch.nn as nn
import torch.nn
import os.path as path


class RGN(nn.Module):

    def __init__(self, d_in, d_hidden, d_out, n_layers):
        super(RGN, self).__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.d_out = d_out
        self.n_layers = n_layers
        self.lstm = nn.LSTM(d_in,
                            d_hidden,
                            n_layers,
                            bidirectional=True,
                            batch_first=True)
        self.hidden2out = nn.Linear(2 * self.hidden_dim, d_out)
        self.device = torch.device("cuda")
        self.hidden2out.bias = nn.Parameter(
            torch.FloatTensor(np.arctanh(self.load_angle_means())))
        nn.init.xavier_normal_(self.hidden2out.weight)

    def init_hidden(self, batch_size):
        """ Initialize the hidden state vectors at the start of a batch iteration. """
        h = torch.zeros(self.num_layers * 2, batch_size, self.d_hidden).to(self.device)
        c = torch.zeros(self.num_layers * 2, batch_size, self.d_hidden).to(self.device)
        return h, c

    def forward(self, sequence, lengths):
        h, c = self.init_hidden(len(lengths))
        sequence = nn.utils.rnn.pack_padded_sequence(sequence, lengths, batch_first=True)
        output, (h, c) = self.lstm(sequence, (h, c))
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(output,
                                                                  batch_first=True)
        output = self.hidden2out(output)
        output = output.view(output.shape[0], output.shape[1], self.d_out)
        return output

    def load_angle_means(self):
        """
        Loads the average angle vector in order to initialize the bias out the output layer. This allows the model to
        begin predicting the average angle vectors and must only learn to predict the difference.
        """
        data, ext = path.splitext(self.data_path)
        angle_mean_path = data + "_mean.npy"
        if not path.exists(angle_mean_path):
            angle_mean_path_new = "protein/190602_query4_mean.npy"
            print(
                f"[Info] Unable to find {angle_mean_path}. Loading angle means from {angle_mean_path_new} instead."
            )
            angle_mean_path = angle_mean_path_new
        angle_means = np.load(angle_mean_path)
        return angle_means
