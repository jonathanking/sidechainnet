import torch
import sidechainnet as scn

from sidechainnet.data_handlers import SCNDataset, SCNProtein
from sidechainnet.utils.openmm import OpenMMEnergy


def test_add_hydrogen_numpy():
    d = scn.load("debug", scn_dir="/home/jok120/openmm_loss/sidechainnet_data")
    d = SCNDataset(d)
    p = d['40#2BDS_1_A']  # Starts with 2 Alanines
    p.coords = p.coords[:28, :]
    p.seq = p.seq[:2]
    p.add_hydrogens()


def test_add_hydrogen_torch():
    d = scn.load("debug", scn_dir="/home/jok120/openmm_loss/sidechainnet_data")
    d = SCNDataset(d)
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
