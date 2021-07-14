import torch
import sidechainnet as scn

from sidechainnet.data_handlers import SCNDataset, SCNProtein

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
    d = SCNDataset(scn.load("debug",
                            scn_dir="/home/jok120/openmm_loss/sidechainnet_data"))
    ex = d["40#2BDS_1_A"]
    print(ex.get_energy_difference())
    print(ex.get_rmsd_difference())
