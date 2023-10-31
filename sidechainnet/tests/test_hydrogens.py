import numpy as np
import torch
from torch.autograd.functional import jacobian as get_jacobian
from torch.autograd import gradcheck

import cProfile
from pstats import Stats, SortKey
from tqdm import tqdm

import sidechainnet as scn
from sidechainnet.utils.openmm_loss import OpenMMEnergy, OpenMMEnergyH
from sidechainnet.examples import get_alphabet_protein

torch.autograd.set_detect_anomaly(True)
torch.set_printoptions(sci_mode=False, precision=3)
np.set_printoptions(precision=3)


def test_add_hydrogen_numpy():
    p = get_alphabet_protein()
    p.numpy()
    p.add_hydrogens()
    assert not torch.is_tensor(p.hcoords)


def test_add_hydrogen_torch():
    p = get_alphabet_protein()
    p.torch()
    p.add_hydrogens()
    assert torch.is_tensor(p.hcoords)


def test_energy_backwards():
    p = get_alphabet_protein()
    p.torch()
    p.add_hydrogens(force_requires_grad=True)
    print(p.hcoords.requires_grad)
    energy_loss = OpenMMEnergyH()
    e = energy_loss.apply(p, p.hcoords)
    print(e)
    e.backward()
