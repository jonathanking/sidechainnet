from __future__ import unicode_literals
from io import StringIO

import os
from distutils import dir_util

import numpy as np
import torch
import prody
from pytest import fixture
import pytest
from tqdm import tqdm

import sidechainnet as scn
import sidechainnet.utils.openmm_loss as mm
from sidechainnet.examples import get_alphabet_protein
from sidechainnet.utils.openmm_loss import OpenMMEnergy, OpenMMEnergyH

torch.autograd.set_detect_anomaly(True)
torch.set_printoptions(sci_mode=False, precision=3)
# Turn off scientific notation for numpy
np.set_printoptions(suppress=True)

SCN_DIR = '/home/jok120/scn221001/'
TOLERANCE = 1e-3


def SCNProtein_from_str(some_string, pdbid=""):
    sio = StringIO(some_string)
    # First, use Prody to parse the PDB file
    chain = prody.parsePDBStream(sio)
    # Next, use SidechainNet to make the relevant measurements given the Prody chain obj
    (dihedrals_np, coords_np, observed_sequence, unmodified_sequence,
     is_nonstd) = scn.utils.measure.get_seq_coords_and_angles(chain, replace_nonstd=True)
    scndata = {
        "coordinates": coords_np.reshape(len(observed_sequence), -1, 3),
        "angles": dihedrals_np,
        "sequence": observed_sequence,
        "unmodified_seq": unmodified_sequence,
        "mask": "+" * len(observed_sequence),
        "is_modified": is_nonstd,
        "id": pdbid,
    }
    return scn.SCNProtein(**scndata)


@fixture
def af2pred_str():
    return """REMARK no_recycling=3, max_templates=4, config_preset=model_1_ptm
PARENT 3n56_D 1q01_A 7bri_L 4qlp_B
ATOM      1  N   SER A   1      26.662  -8.986 -32.998  1.00 36.72           N  
ATOM      2  CA  SER A   1      25.725  -8.730 -34.087  1.00 36.72           C  
ATOM      3  C   SER A   1      25.856  -7.301 -34.603  1.00 36.72           C  
ATOM      4  CB  SER A   1      24.289  -8.986 -33.631  1.00 36.72           C  
ATOM      5  O   SER A   1      25.902  -6.353 -33.817  1.00 36.72           O  
ATOM      6  OG  SER A   1      24.021 -10.376 -33.569  1.00 36.72           O  
ATOM      7  N   PRO A   2      44.188 -25.808 -29.406  1.00 35.73           N  
ATOM      8  CA  PRO A   2      45.355 -26.650 -29.676  1.00 35.73           C  
ATOM      9  C   PRO A   2      45.218 -28.054 -29.093  1.00 35.73           C  
ATOM     10  CB  PRO A   2      46.501 -25.890 -29.004  1.00 35.73           C  
ATOM     11  O   PRO A   2      45.505 -29.041 -29.775  1.00 35.73           O  
ATOM     12  CG  PRO A   2      45.842 -24.742 -28.308  1.00 35.73           C  
ATOM     13  CD  PRO A   2      44.371 -24.786 -28.606  1.00 35.73           C  
TER      14      PRO A   2
END
"""


@fixture
def af2pred_protein(af2pred_str):
    return SCNProtein_from_str(af2pred_str, pdbid="EXMP")


@fixture
def af2pred_protein2():
    return scn.SCNProtein.from_pdb("/home/jok120/sidechainnet/sidechainnet/tests/"
                                   "test_openmm/1A92_1_A_model_1_ptm_unrelaxed.pdb")



def test_load_scndir():
    d = scn.load("debug",
                 scn_dir=SCN_DIR,
                 scn_dataset=True,
                 complete_structures_only=True)
    p = get_alphabet_protein()


def test_get_energy_from_pdbfile_fails_nohy(af2pred_protein):
    with pytest.raises(ValueError):
        af2pred_protein.get_energy()


def test_get_energy_from_pdbfile_addh_via_fastbuild(af2pred_str, tmp_path):
    with open(tmp_path / "af2pred.pdb", "w") as f:
        f.write(af2pred_str)
    p = scn.SCNProtein.from_pdb(tmp_path / "af2pred.pdb")
    p.fastbuild(add_hydrogens=True, inplace=True)
    p.get_energy()


def test_get_energy_from_af2pred2(af2pred_protein2):
    af2pred_protein2.fastbuild(add_hydrogens=True, inplace=True)
    af2pred_protein2.get_energy()


def test_get_energy_from_alphabet_addh_via_fastbuild():
    p = get_alphabet_protein()
    p.fastbuild(add_hydrogens=True, inplace=True)
    p.get_energy()


def test_gradcheck_small():
    torch.cuda.synchronize()
    p = get_alphabet_protein()
    p.cuda()
    p.seq = p.seq[1:3]
    p.angles = p.angles[1:3]
    p.angles = torch.tensor(p.angles, dtype=torch.float64, requires_grad=True)

    p.fastbuild(add_hydrogens=True, inplace=True)

    openmmf = OpenMMEnergyH()
    _input = p, p.hcoords, 1, 1e14
    test = torch.autograd.gradcheck(
        openmmf.apply,
        _input,
        check_undefined_grad=True,
        raise_exception=False,
        rtol=.001,
        eps=1e-3)
    assert test


def test_gradcheck_large():
    torch.cuda.synchronize()
    p = get_alphabet_protein()
    p.cuda()
    # p.seq = p.seq[1:10]
    # p.angles = p.angles[1:10]
    p.angles = torch.tensor(p.angles, dtype=torch.float64, requires_grad=True)

    p.fastbuild(add_hydrogens=True, inplace=True)

    openmmf = OpenMMEnergyH()
    _input = p, p.hcoords, 1, 1e14
    test = torch.autograd.gradcheck(
        openmmf.apply,
        _input,
        check_undefined_grad=True,
        raise_exception=True,
        rtol=.1,
        eps=1e-3)
    assert test


def get_af2pred_protein2():
    return scn.SCNProtein.from_pdb("/home/jok120/sidechainnet/sidechainnet/tests/"
                                   "test_openmm/1A92_1_A_model_1_ptm_unrelaxed.pdb")


@pytest.mark.parametrize("protein", [get_alphabet_protein(), get_af2pred_protein2()])
def test_similar_energy1(protein):
    """Assert that a protein's energy is similar when measured via OpenMM and eloss."""
    protein = get_alphabet_protein()
    protein.fastbuild(add_hydrogens=True, inplace=True)

    # First test if the energy is similar when using the same openmm representation
    protein.initialize_openmm()
    eloss_fn = OpenMMEnergyH()
    eloss = eloss_fn.apply(protein, protein.hcoords)

    e = protein.get_energy(return_unitless_kjmol=True)
    assert torch.abs(eloss - e) < TOLERANCE


@pytest.mark.parametrize("protein", [get_alphabet_protein(), get_af2pred_protein2()])
def test_similar_energy2(protein):
    protein = get_alphabet_protein()
    protein.fastbuild(add_hydrogens=True, inplace=True)

    # Test that the above works when computed in reverse
    protein.initialize_openmm()
    e = protein.get_energy(return_unitless_kjmol=True)

    eloss_fn = OpenMMEnergyH()
    eloss = eloss_fn.apply(protein, protein.hcoords)
    assert torch.abs(eloss - e) < TOLERANCE

@pytest.mark.parametrize("protein", [get_alphabet_protein(), get_af2pred_protein2()])
def test_similar_energy3(protein):
    protein = get_alphabet_protein()
    protein.fastbuild(add_hydrogens=True, inplace=True)

    # Test that you don't need to init openmm explicitly
    eloss_fn = OpenMMEnergyH()
    eloss = eloss_fn.apply(protein, protein.hcoords)

    e = protein.get_energy(return_unitless_kjmol=True)
    assert torch.abs(eloss - e) < TOLERANCE

@pytest.mark.parametrize("protein", [get_alphabet_protein(), get_af2pred_protein2()])
def test_similar_energy4(protein):
    protein = get_alphabet_protein()
    protein.fastbuild(add_hydrogens=True, inplace=True)

    # Now test if the energy is similar when resetting the openmm representation
    protein.initialize_openmm()
    eloss_fn = OpenMMEnergyH()
    eloss = eloss_fn.apply(protein, protein.hcoords)

    protein.openmm_initialized = False
    protein.initialize_openmm()
    e = protein.get_energy(return_unitless_kjmol=True)
    assert torch.abs(eloss - e) < TOLERANCE


def test_gradcheck2():
    protein = get_alphabet_protein()
    protein.fastbuild(add_hydrogens=True, inplace=True)
    hcoords = torch.tensor(protein.hcoords, dtype=torch.float64, requires_grad=True)
    eps = 1e-5
    energy_loss = mm.OpenMMEnergyH()
    energy = energy_loss.apply
    c1, c2 = torch.tensor(hcoords.clone().detach()), torch.tensor(
        hcoords.clone().detach())
    c1[0, 0] += eps
    c2[0, 0] -= eps
    f1 = energy(protein, c1)
    f2 = energy(protein, c2)
    dfdx = (f1 - f2) / 2 * eps

    analytical_energy = mm.OpenMMEnergyH()
    analytical_energy = analytical_energy.apply(protein, hcoords)
    analytical_energy.backward()
    forces = hcoords.grad
    force = forces[0, 0]

    difference = np.float64(torch.abs(dfdx - force)) / max(np.float64(torch.abs(dfdx)),
                                                           np.float64(torch.abs(force)))
    return difference


def test_hydrogen_partners():
    from sidechainnet.structure.HydrogenBuilder import (HYDROGEN_NAMES, HYDROGEN_PARTNERS)

    for resname in HYDROGEN_NAMES.keys():
        assert len(HYDROGEN_NAMES[resname]) == len(HYDROGEN_PARTNERS[resname])
        for atomname in HYDROGEN_PARTNERS[resname]:
            assert not atomname.startswith("H")


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


### Deprecated ###

# def test_pytorch_layer():
#     # Fails because add_hydrogens is not yet implemented
#     d = scn.load("debug", scn_dir=SCN_DIR, scn_dataset=True)
#     p = d["40#2BDS_1_A"]
#     p.add_hydrogens()  # Must be done at every step of optimization
#     p.initialize_openmm()  # Need only be done once, but does need to see hydrogens
#     energy = p.get_energy()
#     print(energy)
#     assert energy == p.get_energy()
#     p.add_hydrogens()
#     assert energy == p.get_energy()

# def test_pytorch_layer2():
#     # Setup
#     d = scn.load("debug", scn_dir=SCN_DIR, scn_dataset=True)
#     p = d[0]
#     p.cuda()
#     p.angles = torch.tensor(p.angles, requires_grad=True)
#     p.fast_build(from_angles=True)
#     p.coords = torch.tensor(p.coords, requires_grad=True)
#     loss_layer = mm.OpenMMEnergy()
#     opt = optim.SGD([p.coords], lr=1e-3)

#     for i in range(10):
#         energy = loss_layer.apply(p)
#         print(energy)
#         energy.requires_grad = True
#         energy.backward()
#         opt.step()

# def test_add_hydrogens_to_struct_with_existing_h():
#     d = scn.load("debug",
#                  scn_dir=SCN_DIR,
#                  scn_dataset=True)
#     p = d["40#2BDS_1_A"]
#     p.add_hydrogens()  # Must be done at every step of optimization
#     p.initialize_openmm()  # Need only be done once, but does need to see hydrogens
#     energy = p.get_energy()
#     p.add_hydrogens()
#     assert energy == p.get_energy()

# def test_minimize_scndataset():
#     d = scn.load("debug",
#                  scn_dir=SCN_DIR,
#                  scn_dataset=True,
#                  complete_structures_only=True)
#     p = d[0]
#     p.add_hydrogens()
#     p.minimize()

#

# def test_gradcheck2():
#     # Load data
#     d = scn.load("debug",
#                  scn_dir=SCN_DIR,
#                  scn_dataset=True)
#     p = d[0]

#     # Truncate to 2 AAs
#     p.seq = p.seq[:2]
#     p.hcoords = p.hcoords[:14 * 2]
#     p.angles = torch.tensor(p.angles[:12 * 2], requires_grad=True)
#     p.coords = torch.tensor(p.coords[:14 * 2], requires_grad=True)

#     dfdx, forces = _gradcheck(p, p.coords)

# def test_gradcheck_hsum():
#     # Load data
#     d = scn.load("debug", scn_dir=SCN_DIR, scn_dataset=True)
#     p = d["1HD1_1_A"]

#     # Truncate to 2 AAs
#     p.seq = p.seq[:2]
#     p.hcoords = p.hcoords[:14 * 2]
#     p.angles = torch.tensor(p.angles[:12 * 2], requires_grad=True)
#     p.coords = torch.tensor(p.coords[:14 * 2], requires_grad=True)

#     energy_loss = OpenMMEnergy()
#     opt = torch.optim.LBFGS([p.coords], lr=1e-7)

#     losses = []

#     for i in tqdm(range(50)):

#         def closure():
#             opt.zero_grad()
#             loss = energy_loss.apply(p, p.coords)
#             loss.backward()
#             print(loss)
#             losses.append(float(loss.detach().numpy()))
#             return loss

#         opt.step(closure)