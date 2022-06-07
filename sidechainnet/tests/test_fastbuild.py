"""Test the fastbuild coordinate building module."""
import pytest
import torch

import sidechainnet as scn
from sidechainnet.dataloaders.SCNProtein import SCNProtein

@pytest.fixture
def p():
    nan = torch.nan
    angles = torch.tensor([
        [    nan,  2.9864,  3.1411,  1.8979,  2.0560,  2.1353, -1.1941, -2.9177, -0.0954,     nan,     nan,     nan],
        [-2.2205, -2.7720,  3.1174,  1.9913,  2.0165,  2.1277,  1.9315,  1.3732, 2.8204, -1.2943,     nan,     nan],
        [-1.8046,  2.6789, -3.0887,  1.8948,  2.0197,  2.1358,  2.3525, -2.9406, -1.0214,     nan,     nan,     nan],
        [-2.6590,  2.7039,  3.0630,  1.9808,  2.0562,  2.1439,  1.5275,  3.0964, 1.1357, -3.0276,     nan,     nan],
        [-2.6066, -0.7829,     nan,  1.9416,     nan,     nan,  1.5148, -3.0244, -2.8980, -0.4080,     nan,     nan]], dtype=torch.float64)
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
                 scn_dataset=True)
    plarge = d[-10]
    plarge.torch()
    print("Loaded", plarge)
    return plarge


def test_build01(p: SCNProtein):
    """Build a simple protein from angles, excluding hydrogens."""
    new_coords = p.fastbuild(add_hydrogens=False)
    assert new_coords.numpy() == pytest.approx(p.coords.numpy())


def test_build_large01(plarge: SCNProtein):
    """Build a large protein from angles, excluding hydrogens."""
    new_coords = plarge.fastbuild(add_hydrogens=False)
    assert new_coords.numpy() == pytest.approx(plarge.coords.numpy())
