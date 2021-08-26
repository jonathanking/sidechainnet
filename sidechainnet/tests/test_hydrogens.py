import torch
from torch.autograd.functional import jacobian


import sidechainnet as scn
from sidechainnet.utils.openmm import OpenMMEnergy


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
    p.seq = p.seq[start:start+l]
    p.coords = p.coords[start*14:start*14 + 14*l]
    p.coords = torch.tensor(p.coords, dtype=torch.float64, requires_grad=True)
    return p


def test_26atom_rep():
    p = load_p(38, 2)  # includes 6 primitives (RS)
    p.add_hydrogens()  # Debug this step - any non-torch primitives?
    add_h = get_alias(p)
    j = jacobian(add_h, p.coords)

