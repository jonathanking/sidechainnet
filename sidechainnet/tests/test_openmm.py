import sidechainnet as scn
import torch
import sidechainnet.utils.openmm as mm
import torch.optim as optim
import numpy as np


def test_pytorch_layer():
    d = scn.load("debug",
                 scn_dir="/home/jok120/openmm_loss/sidechainnet_data",
                 scn_dataset=True)
    p = d["40#2BDS_1_A"]
    p.add_hydrogens()      # Must be done at every step of optimization
    p.initialize_openmm()  # Need only be done once, but does need to see hydrogens
    energy = p.get_energy()
    print(energy)
    assert energy == p.get_energy()
    p.add_hydrogens()
    assert energy == p.get_energy()


def test_pytorch_layer2():
    # Setup
    d = scn.load("debug",
                 scn_dir="/home/jok120/openmm_loss/sidechainnet_data",
                 scn_dataset=True)
    p = d["1HD1_1_A"]
    p.angles = torch.tensor(p.angles, requires_grad=True)
    p.add_hydrogens(from_angles=True)
    p.coords = torch.tensor(p.coords, requires_grad=True)
    loss_layer = mm.OpenMMEnergy()
    opt = optim.SGD([p.coords], lr=1e-3)

    for i in range(10):
        energy = loss_layer.apply(p)
        print(energy)
        energy.requires_grad = True
        energy.backward()
        opt.step()


def test_add_hydrogens_to_struct_with_existing_h():
    d = scn.load("debug",
                 scn_dir="/home/jok120/openmm_loss/sidechainnet_data",
                 scn_dataset=True)
    p = d["40#2BDS_1_A"]
    p.add_hydrogens()  # Must be done at every step of optimization
    p.initialize_openmm()  # Need only be done once, but does need to see hydrogens
    energy = p.get_energy()
    p.add_hydrogens()
    assert energy == p.get_energy()


def gradcheck(protein, coords, eps=1e-5):
    energy_loss = mm.OpenMMEnergy()
    energy = energy_loss.apply
    c1, c2 = torch.tensor(coords.clone().detach()), torch.tensor(coords.clone().detach())
    c1[0,0] += eps
    c2[0,0] -= eps
    f1, _ = energy(protein, c1)
    f2, _ = energy(protein, c2)
    dfdx = (f1 - f2) / 2 * eps

    analytical_energy = mm.OpenMMEnergy()
    analytical_energy, forces = analytical_energy.apply(protein, coords)
    force = -forces[0,0]

    difference = np.float64(torch.abs(dfdx-force)) / max(np.float64(torch.abs(dfdx)), np.float64(torch.abs(force)))
    return difference

def test_gradcheck():
    # Load data
    d = scn.load("debug", scn_dir="/home/jok120/openmm_loss/sidechainnet_data", scn_dataset=True)
    p = d["1HD1_1_A"]

    # Truncate to 2 AAs
    p.seq = p.seq[:2]
    p.hcoords = p.hcoords[:14*2]
    p.angles = torch.tensor(p.angles[:12*2], requires_grad=True)
    p.coords = torch.tensor(p.coords[:14*2], requires_grad=True)


    dfdx, forces = gradcheck(p, p.coords)
