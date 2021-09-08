import numpy as np
import torch
from torch.autograd.functional import jacobian as get_jacobian
from torch.autograd import gradcheck

import sidechainnet as scn
from sidechainnet.utils.openmm import OpenMMEnergy

torch.autograd.set_detect_anomaly(True)
torch.set_printoptions(sci_mode=False, precision=3)
np.set_printoptions(precision=3)


def test_add_hydrogen_numpy():
    d = scn.load("debug",
                 scn_dir="/home/jok120/openmm_loss/sidechainnet_data",
                 scn_dataset=True)
    p = d['40#2BDS_1_A']  # Starts with 2 Alanines
    p.coords = p.coords[:28, :]
    p.seq = p.seq[:2]
    p.add_hydrogens()


def test_add_hydrogen_torch():
    d = scn.load("debug",
                 scn_dir="/home/jok120/openmm_loss/sidechainnet_data",
                 scn_dataset=True)
    p = d['40#2BDS_1_A']  # Starts with 2 Alanines
    p.coords = torch.tensor(p.coords[:28, :])
    p.seq = p.seq[:2]
    p.add_hydrogens()


def test_nterminal():
    d = scn.load("debug",
                 scn_dir="/home/jok120/openmm_loss/sidechainnet_data",
                 scn_dataset=True)
    ex = d["40#2BDS_1_A"]
    print(ex.get_energy_difference())
    print(ex.get_rmsd_difference())


def test_energy_backwards():
    d = scn.load("debug",
                 scn_dir="/home/jok120/openmm_loss/sidechainnet_data",
                 scn_dataset=True)
    p = d["1HD1_1_A"]
    p.angles = torch.tensor(p.angles, requires_grad=True)
    p.coords = torch.tensor(p.coords, requires_grad=True)
    energy_loss = OpenMMEnergy()
    e = energy_loss.apply(p, p.coords)
    print(e)
    e.backward()


def get_alias(protein):
    def add_h(tcoords):
        """Mimics f(X) -> X_H."""
        protein.add_hydrogens(coords=tcoords)
        return protein.hcoords
    return add_h


def load_p(start=0, l=2):
    d = scn.load("debug", scn_dir="/home/jok120/openmm_loss/sidechainnet_data", scn_dataset=True)
    p = d["1HD1_1_A"]
    if l > 0:
        p.seq = p.seq[start:start+l]
        p.coords = p.coords[start*14:start*14 + 14*l]
    p.coords = torch.tensor(p.coords, dtype=torch.float64, requires_grad=True)
    return p


def test_26atom_rep():
    p = load_p(38, 2)  # includes 6 primitives (RS)
    p.add_hydrogens()  # Debug this step - any non-torch primitives?
    add_h = get_alias(p)
    j = get_jacobian(add_h, p.coords)


# OpenMM Gradients
def test_openmm_gradcheck():
    openmmf = OpenMMEnergy()
    for length in range(2, 6):
        for starting_idx in range(75-length):
            p = load_p(start=starting_idx, l=2)
            j = get_jacobian(get_alias(p), p.coords)
            _input = p, p.coords, "all", j
            test = gradcheck(openmmf.apply, _input, check_undefined_grad=True)
            print(starting_idx, test)


def test_openmm_gradcheck2():
    openmmf = OpenMMEnergy()
    p = load_p(start=38, l=10)
    j = get_jacobian(get_alias(p), p.coords)
    _input = p, p.coords, "all", j
    test = gradcheck(openmmf.apply, _input, check_undefined_grad=True)
    print(test)


def test_openmm_gradcheck3():
    openmmf = OpenMMEnergy()
    p = load_p(start=0, l=-1)
    j = get_jacobian(get_alias(p), p.coords)
    _input = p, p.coords, "all", j
    test = gradcheck(openmmf.apply, _input, check_undefined_grad=True)
    print(test)
