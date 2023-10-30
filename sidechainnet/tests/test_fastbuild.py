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
    # 1st line uses the new measurement methodology for X0
    # 2nd line below uses default tetrahedral geom to place CB atom with X0
    angles = torch.tensor([
        # [nan, 2.9864, 3.1411, 1.8979, 2.0560, 2.1353, 0.9887, -2.9177, -0.0954, nan, nan, nan],
        [nan, 2.9864, 3.1411, 1.8979, 2.0560, 2.1353, np.deg2rad(-109.5), -2.9177, -0.0954, nan, nan, nan],
        [-2.2205, -2.7720, 3.1174, 1.9913, 2.0165, 2.1277, 1.9315, 1.3732, 2.8204, -1.2943, nan, nan],
        [-1.8046, 2.6789, -3.0887, 1.8948, 2.0197, 2.1358, 2.3525, -2.9406, -1.0214, nan, nan, nan],
        [-2.6590, 2.7039, 3.0630, 1.9808, 2.0562, 2.1439, 1.5275, 3.0964, 1.1357, -3.0276, nan, nan],
        [-2.6066, -0.7829, nan, 1.9416, nan, nan, 1.5148, -3.0244, -2.8980, -0.4080, nan, nan]
        ], dtype=torch.float64)
    coordinates = torch.tensor([
        [  9.1920,  64.2680,   3.3370],
        [ 10.0050,  64.7540,   2.1820],
        [  9.9800,  63.6780,   1.0680],
        [  9.4220,  62.6010,   1.2670],
        [ 11.4280,  65.0700,   2.6870],
        [ 12.3400,  65.8630,   1.7450],
        [ 12.0780,  66.2460,   0.4450],
        [ 13.1990,  66.8160,  -0.1220],
        [ 14.2060,  66.8290,   0.8060],
        [ 15.5160,  67.3040,   0.7030],
        [ 16.3170,  67.1960,   1.8120],
        [ 15.8440,  66.6310,   3.0130],
        [ 14.5450,  66.1570,   3.1180],
        [ 13.7010,  66.2500,   2.0000],
        [ 10.4980,  63.9990,  -0.1190],
        [ 10.5270,  63.0540,  -1.2470],
        [ 11.9230,  62.8670,  -1.8760],
        [ 12.9590,  63.1150,  -1.2480],
        [  9.5600,  63.5110,  -2.3470],
        [ 10.0530,  64.6440,  -3.2600],
        [  9.2350,  64.6880,  -4.5400],
        [ 10.0250,  65.9760,  -2.5330],
        [     nan,      nan,      nan],
        [     nan,      nan,      nan],
        [     nan,      nan,      nan],
        [     nan,      nan,      nan],
        [     nan,      nan,      nan],
        [     nan,      nan,      nan],
        [ 11.9330,  62.3970,  -3.1190],
        [ 13.1700,  62.1960,  -3.8540],
        [ 13.3270,  63.3330,  -4.8620],
        [ 12.3410,  63.8710,  -5.3730],
        [ 13.1600,  60.8410,  -4.5590],
        [ 14.5210,  60.4570,  -5.1070],
        [ 15.0780,  61.2080,  -5.9300],
        [ 15.0360,  59.3880,  -4.7290],
        [     nan,      nan,      nan],
        [     nan,      nan,      nan],
        [     nan,      nan,      nan],
        [     nan,      nan,      nan],
        [     nan,      nan,      nan],
        [     nan,      nan,      nan],
        [ 14.5730,  63.6250,  -5.2110],
        [ 14.8940,  64.7260,  -6.1130],
        [ 16.2100,  64.5090,  -6.9120],
        [ 17.0660,  63.7270,  -6.4870],
        [ 15.0310,  65.9650,  -5.2130],
        [ 15.4200,  67.3410,  -5.7340],
        [ 14.3060,  67.7890,  -6.6420],
        [ 15.6320,  68.3370,  -4.5770],
        [     nan,      nan,      nan],
        [     nan,      nan,      nan],
        [     nan,      nan,      nan],
        [     nan,      nan,      nan],
        [     nan,      nan,      nan],
        [     nan,      nan,      nan],
        [ 16.3420,  65.1450,  -8.0830],
        [ 17.5860,  65.0960,  -8.8890],
        [ 17.7800,  66.3900,  -9.6920],
        [ 16.7950,  66.8630, -10.2970],
        [ 17.6770,  63.8950,  -9.8520],
        [ 19.0770,  63.8170, -10.5390],
        [ 19.1590,  62.8800, -11.7410],
        [ 18.3380,  61.9500, -11.8410],
        [ 20.0600,  63.0720, -12.5890],
        [     nan,      nan,      nan],
        [     nan,      nan,      nan],
        [     nan,      nan,      nan],
        [     nan,      nan,      nan],
        [     nan,      nan,      nan]], dtype=torch.float64)
    p = SCNProtein(
        coordinates=coordinates,
        angles=angles,
        sequence="WLDLE",
        mask="+++++",
        id="1A38_2_P",
        split="train"
    )
    p.torch()
    return p


@pytest.fixture
def plarge():
    d = scn.load('demo',
                 local_scn_path='/home/jok120/sidechainnet_data/sidechainnet_debug.pkl',
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
    p.numpy()
    pcopy = p.copy()
    built_coords = pcopy.build_coords_from_angles()

    # Remove rows with nans
    built_coords = built_coords[~np.isnan(built_coords).any(axis=1)]
    p.coords = p.coords[~np.isnan(p.coords).any(axis=1)]

    # Remove first 3 atoms which may differ due to assumptions
    built_coords = built_coords[3:, :]
    p.coords = p.coords[3:, :]

    # Compare
    aligned_minimized = pr.calcTransformation(built_coords, p.coords).apply(built_coords)
    rmsd = pr.calcRMSD(aligned_minimized, p.coords)

    assert rmsd <= 1
    assert aligned_minimized == pytest.approx(p.coords, abs=1)


def test_fastbuild_vs_oldbuild01(p: SCNProtein):
    """Build a simple protein from angles, excluding hydrogens."""
    # p.angles[0, 0] = 0
    fast_coords = p.fastbuild(add_hydrogens=False)
    built_coords = p.build_coords_from_angles()
    fast_coords = fast_coords.reshape(-1, 3)

    # Remove first 3 atoms which may differ due to assumptions
    built_coords = built_coords[3:, :].detach().numpy()
    fast_coords = fast_coords[3:, :].detach().numpy()

    # Remove rows with nans
    built_coords = built_coords[~np.isnan(built_coords).any(axis=1)]
    fast_coords = fast_coords[~np.isnan(fast_coords).any(axis=1)]

    # Compare
    aligned_minimized = pr.calcTransformation(fast_coords, built_coords).apply(fast_coords)
    rmsd = pr.calcRMSD(aligned_minimized, built_coords)

    assert aligned_minimized == pytest.approx(built_coords, abs=2)
    assert rmsd <= 1

def test_fasbuildh_01(p: SCNProtein):
    """Build a simple protein from angles, including hydrogens."""
    fast_coords = p.fastbuild(add_hydrogens=True)
    p.coords = p.hcoords = fast_coords
    p.to_pdb("/home/jok120/Downloads/predh36.pdb")


def test_fasbuildh_02(p: SCNProtein):
    """Build a simple protein from angles, including hydrogens. Compute E."""
    fast_coords = p.fastbuild(add_hydrogens=True)
    p.coords = p.hcoords = fast_coords

    p.minimize()


def test_fastbuild_openmmhloss(p: SCNProtein):
    fast_coords = p.fastbuild(add_hydrogens=True)
    to_optim = (p.hcoords).detach().clone().requires_grad_(True)

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

    # TODO To apply gradients to parameters, make sure to initialize
    # OpenMM with bonded terms only. Then do a backwards step and trace

    # the gradients back to the parameters that built the structure.

def test_measure_x0_with_fictitious_atom():
    _init_dssp_data()
    data = process_id("1A38_2_P")
    print(data)

def test_alphabet():
    d = scn.load_pdb("/home/jok120/Downloads/alphabet.pdb", pdbid='ALFA')
    d.torch()
    # coords = d.fastbuild(add_hydrogens=False)
    hcoords = d.fastbuild(add_hydrogens=True)
    d.coords = hcoords
    d.hcoords = hcoords
    d.has_hydrogens = True
    d.sb = None
    d.to_pdb("/home/jok120/Downloads/alfa31.pdb", title='ALFA')
    print(hcoords)


def test_alphabet_build():
    p = get_alphabet_protein()
    # p.to_pdb("./alphabet_build01.pdb")
    # p.fastbuild(inplace=True)
    # p.to_pdb("./alphabet_build02.pdb")
    p.fastbuild(add_hydrogens=True, inplace=True)
    p.to_pdb("/home/jok120/Downloads/build_alfa04.pdb")


def test_heavy_atom_build_info():
    from sidechainnet.structure import fastbuild
    binf_allatom = fastbuild.SC_ALL_ATOM_BUILD_PARAMS
    binf_heavy = fastbuild.SC_HEAVY_ATOM_BUILD_PARAMS
    print("Hello world.s")