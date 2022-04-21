import sidechainnet as scn
import torch
import sidechainnet.utils.openmm as mm
import torch.optim as optim
import numpy as np
from sidechainnet.utils.openmm import OpenMMEnergy, OpenMMEnergyH
from tqdm import tqdm

SCN_DIR = '/home/jok120/sidechainnet_data'

def test_pytorch_layer():
    d = scn.load("debug",
                 scn_dir=SCN_DIR,
                 scn_dataset=True)
    p = d["40#2BDS_1_A"]
    p.add_hydrogens()  # Must be done at every step of optimization
    p.initialize_openmm()  # Need only be done once, but does need to see hydrogens
    energy = p.get_energy()
    print(energy)
    assert energy == p.get_energy()
    p.add_hydrogens()
    assert energy == p.get_energy()


def test_pytorch_layer2():
    # Setup
    d = scn.load("debug",
                 scn_dir=SCN_DIR,
                 scn_dataset=True)
    p = d["1HD1_1_A"]
    p.cuda()
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
                 scn_dir=SCN_DIR,
                 scn_dataset=True)
    p = d["40#2BDS_1_A"]
    p.add_hydrogens()  # Must be done at every step of optimization
    p.initialize_openmm()  # Need only be done once, but does need to see hydrogens
    energy = p.get_energy()
    p.add_hydrogens()
    assert energy == p.get_energy()


def _gradcheck(protein, coords, eps=1e-5):
    energy_loss = mm.OpenMMEnergy()
    energy = energy_loss.apply
    c1, c2 = torch.tensor(coords.clone().detach()), torch.tensor(coords.clone().detach())
    c1[0, 0] += eps
    c2[0, 0] -= eps
    f1, _ = energy(protein, c1)
    f2, _ = energy(protein, c2)
    dfdx = (f1 - f2) / 2 * eps

    analytical_energy = mm.OpenMMEnergy()
    analytical_energy, forces = analytical_energy.apply(protein, coords)
    force = -forces[0, 0]

    difference = np.float64(torch.abs(dfdx - force)) / max(np.float64(torch.abs(dfdx)),
                                                           np.float64(torch.abs(force)))
    return difference


def test_gradcheck():
    # Load data
    d = scn.load("debug",
                 scn_dir=SCN_DIR,
                 scn_dataset=True)
    p = d["1HD1_1_A"]

    # Truncate to 2 AAs
    p.seq = p.seq[:2]
    p.hcoords = p.hcoords[:14 * 2]
    p.angles = torch.tensor(p.angles[:12 * 2], requires_grad=True)
    p.coords = torch.tensor(p.coords[:14 * 2], requires_grad=True)

    dfdx, forces = _gradcheck(p, p.coords)


def test_gradcheck2():
    d = scn.load("debug",
                 scn_dir=SCN_DIR,
                 scn_dataset=True)

    p = d["1HD1_1_A"]
    p.seq = p.seq[:2]
    p.coords = p.coords[:14 * 2]
    p.add_hydrogens()
    p.hcoords = p.hcoords[:24 * 2]
    p.hcoords = torch.tensor(p.hcoords, dtype=torch.float64, requires_grad=True)
    to_optim = torch.tensor(p.hcoords, dtype=torch.float64, requires_grad=True)

    openmmf = OpenMMEnergy()
    _input = p, to_optim, "all", False
    test = torch.autograd.gradcheck(openmmf.apply, _input, check_undefined_grad=True)
    assert test


def test_hydrogen_partners():
    from sidechainnet.structure.HydrogenBuilder import HYDROGEN_NAMES, HYDROGEN_PARTNERS

    for resname in HYDROGEN_NAMES.keys():
        assert len(HYDROGEN_NAMES[resname]) == len(HYDROGEN_PARTNERS[resname])
        for atomname in HYDROGEN_PARTNERS[resname]:
            assert not atomname.startswith("H")


def test_gradcheck_hsum():
    # Load data
    d = scn.load("debug",
                 scn_dir=SCN_DIR,
                 scn_dataset=True)
    p = d["1HD1_1_A"]

    # Truncate to 2 AAs
    p.seq = p.seq[:2]
    p.hcoords = p.hcoords[:14 * 2]
    p.angles = torch.tensor(p.angles[:12 * 2], requires_grad=True)
    p.coords = torch.tensor(p.coords[:14 * 2], requires_grad=True)

    energy_loss = OpenMMEnergy()
    opt = torch.optim.LBFGS([p.coords], lr=1e-7)

    losses = []

    for i in tqdm(range(50)):

        def closure():
            opt.zero_grad()
            loss = energy_loss.apply(p, p.coords)
            loss.backward()
            print(loss)
            losses.append(float(loss.detach().numpy()))
            return loss

        opt.step(closure)


def test_OpenMMEnergyH():
    d = scn.load("debug",
                 scn_dir=SCN_DIR,
                 scn_dataset=True,
                 complete_structures_only=True,
                 filter_by_resolution=True)
    p = d[7]
    p.torch()
    p.cuda()
    p.add_hydrogens()
    # p.get_openmm_repr()
    # p.get_energy_difference()
    # p.get_rmsd_difference()
    # p._make_start_and_end_pdbs()

    to_optim = (p.hcoords).detach().clone().requires_grad_(True)
    starting_coords = to_optim.detach().clone()

    energy_loss = OpenMMEnergyH()
    opt = torch.optim.SGD([to_optim], lr=1e-5)

    losses = []
    coordinates = []

    for i in tqdm(range(10000)):
        coordinates.append(to_optim.detach().cpu().numpy())
        opt.zero_grad()
        loss = energy_loss.apply(p, to_optim)
        loss.backward()
        lossnp = float(loss.detach().cpu().numpy())
        losses.append(lossnp)
        fn = torch.norm(to_optim.grad)
        opt.step()


def test_minimize_scndataset():
    d = scn.load("debug",
                 scn_dir="/home/jok120/sidechainnet_data",
                 scn_dataset=True,
                 complete_structures_only=True)
    p = d[0]
    p.add_hydrogens()
    p.minimize()
