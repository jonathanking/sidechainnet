"""Test the fastbuild coordinate building module."""
from re import A
import numpy as np
import prody as pr
import pytest
import torch
from tqdm import tqdm

import sidechainnet as scn
from sidechainnet.dataloaders.SCNProtein import SCNProtein
from sidechainnet.structure.build_info import NUM_COORDS_PER_RES
from sidechainnet.utils.download import _init_dssp_data, process_id
from sidechainnet.utils.openmm_loss import OpenMMEnergyH
from sidechainnet.examples import get_alphabet_protein

torch.set_printoptions(sci_mode=False)
nan = torch.nan


@pytest.fixture
def p():
    return get_p()


def get_p():
    """Create and return a length-5 example protein, 'WLDLE'."""
    return get_alphabet_protein()


@pytest.fixture
def plarge():
    d = scn.load('debug',
                 scn_dataset=True,
                 complete_structures_only=True)
    plarge = d[-10]
    plarge.torch()
    print("Loaded", plarge)
    return plarge


@pytest.fixture
def palpha():
    return get_alphabet_protein()

def test_angle_to_coord_rebuild_works_for_fixtures(p: SCNProtein):
    """Make sure that rebuilding the protein's coords from its angles is equal."""
    p = get_alphabet_protein()
    p.numpy()
    pcopy = p.copy()
    built_coords = pcopy.fastbuild(add_hydrogens=False).detach().numpy()

    # Flatten and remove rows with nans
    built_coords = built_coords.reshape(-1, 3)
    p.coords = p.coords.reshape(-1, 3)
    built_coords = built_coords[~np.isnan(built_coords).any(axis=1)]
    p.coords = p.coords[~np.isnan(p.coords).any(axis=1)]

    # Remove first 3 atoms which may differ due to assumptions
    built_coords = built_coords[3:, :]
    p.coords = p.coords[3:, :]

    # Compare
    aligned_minimized = pr.calcTransformation(built_coords, p.coords).apply(built_coords)
    rmsd = pr.calcRMSD(aligned_minimized, p.coords)

    assert rmsd <= 1
    assert aligned_minimized == pytest.approx(p.coords, abs=2)


def test_fasbuildh_01(p: SCNProtein):
    """Build a simple protein from angles, including hydrogens."""
    fast_coords = p.fastbuild(add_hydrogens=True)
    p.coords = p.hcoords = fast_coords
    p.to_pdb("predh36.pdb")


def test_fasbuildh_02(p: SCNProtein):
    """Build a simple protein from angles, including hydrogens. Compute E."""
    # p.fastbuild(add_hydrogens=True, inplace=True)
    p = get_alphabet_protein()
    d = scn.load(casp_version=12, casp_thinning='scnmin')
    p = d[0]
    p.torch()
    p.add_hydrogens()
    p.minimize()


def test_fastbuild_openmmhloss(p: SCNProtein):
    p.fastbuild(add_hydrogens=True, inplace=True)
    p.cpu()
    to_optim = (p.hcoords).detach().clone().requires_grad_(True)

    energy_loss = OpenMMEnergyH()
    opt = torch.optim.SGD([to_optim], lr=1e-5)

    losses = []
    coordinates = []

    for i in tqdm(range(10)):
        coordinates.append(to_optim.detach().cpu().numpy())
        opt.zero_grad()
        loss = energy_loss.apply(p, to_optim).cpu()
        loss.backward()
        lossnp = float(loss.detach().cpu().numpy())
        losses.append(lossnp)
        fn = torch.norm(to_optim.grad)
        opt.step()

    # TODO To apply gradients to parameters, make sure to initialize
    # OpenMM with bonded terms only. Then do a backwards step and trace

    # the gradients back to the parameters that built the structure.

def test_measure_x0_with_fictitious_atom():
    _init_dssp_data()
    data = process_id("1A38_2_P")
    print(data)

def test_alphabet():
    d = get_alphabet_protein()
    d.torch()
    # coords = d.fastbuild(add_hydrogens=False)
    hcoords = d.fastbuild(add_hydrogens=True)
    d.coords = hcoords
    d.hcoords = hcoords
    d.has_hydrogens = True
    d.sb = None
    d.to_pdb("alfa31.pdb", title='ALFA')
    print(hcoords)


def test_alphabet_build():
    p = get_alphabet_protein()
    # p.to_pdb("./alphabet_build01.pdb")
    # p.fastbuild(inplace=True)
    # p.to_pdb("./alphabet_build02.pdb")
    p.fastbuild(add_hydrogens=True, inplace=True)
    p.to_pdb("build_alfa04.pdb")


def test_heavy_atom_build_info():
    from sidechainnet.structure import fastbuild
    binf_allatom = fastbuild.SC_ALL_ATOM_BUILD_PARAMS
    binf_heavy = fastbuild.SC_HEAVY_ATOM_BUILD_PARAMS
    print("Hello world.")