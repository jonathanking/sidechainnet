import sidechainnet as scn
import torch
import sidechainnet.utils.openmm as mm
import torch.optim as optim


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
