import torch
import sidechainnet as scn


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
