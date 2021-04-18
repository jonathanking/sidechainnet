import torch
import torch.nn as nn
import sidechainnet as scn
from .openmmfunction import OpenMMFunction
import random
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class OpenMMLayer(nn.Module):

    def __init__(self, sequence, coords):
        super(OpenMMLayer, self).__init__()
        self.seq = sequence
        self.coords = torch.tensor(coords, dtype=torch.float64)
        self.coords = nn.Parameter(self.coords)
        self.coord_mask = np.zeros_like(coords)
        self.coord_mask[self.coords > 0.0] = 1.0

    def forward(self):
        return OpenMMFunction.apply(self.coords, self.seq)

    def step(self, lr, clip_value=.1):
        self.coords.grad.data.clamp_(min=-clip_value, max=clip_value)
        self.coords.data -= lr * self.coords.grad.data
        self.coords.data *= self.coord_mask
        self.coords.grad.data.zero_()


def inject_noise(coords, scale=1):
    """
    :param coords: The co-ordinates of sidechainnet. Dimension L x 14 x 3, where L is the number of residues
    :return: The co-ordinates altered with random noise. All zero co-ordinates means missing atoms. Those are not altered.
    """
    new = np.array(coords)
    nonzero = np.nonzero(new)
    for i in range(len(nonzero[0])):
        new[nonzero[0][i]][nonzero[1][i]] += random.random() * scale
    return new


if __name__ == '__main__':
    data = scn.load(casp_version=12, thinning=30)
    coords = data['train']['crd'][0]
    seq = data['train']['seq'][0]
    coords = inject_noise(coords)
    lr = 0.001
    openmmlayer = OpenMMLayer(seq, coords)
    for i in range(30):
        random.seed(10)
        loss = openmmlayer()
        loss.backward()
        openmmlayer.step(lr)
        print(loss)
