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


def alphabet_protein():
    # yapf: disable
    angles = torch.tensor([[        nan,      1.6403,     -3.0477,      1.9385,      2.1769, 2.2861,      1.0008,         nan,         nan,         nan, nan,         nan],
                 [    -1.3651,      2.2134,      3.0826,      1.9368,      1.9941, 2.0703,      2.7867,     -3.0784,         nan,         nan, nan,         nan],
                 [    -1.4283,      2.9721,     -3.1141,      1.9394,      2.0664, 2.2493,      2.7162,      1.0667,      1.1696,         nan, nan,         nan],
                 [    -0.9306,     -0.1864,     -3.0546,      1.9408,      2.0347, 2.0917,     -3.0698,     -1.2037,     -3.0527,      1.5975, nan,         nan],
                 [    -1.9971,      0.1493,      3.0532,      1.9344,      2.0506, 2.1589,      2.1516,     -1.1137,     -1.5357,         nan, nan,         nan],
                 [     1.4092,      0.2720,      3.0695,      1.9752,      2.0432, 2.0973,         nan,         nan,         nan,         nan, nan,         nan],
                 [    -1.4939,      2.3153,      3.0944,      1.9400,      2.0253, 2.0765,      2.6519,     -1.0513,      2.0782,         nan, nan,         nan],
                 [    -1.6848,      1.9562,      3.0740,      1.9156,      2.0072, 2.0953,      2.4615,     -1.0092,      2.9504,      3.1247, nan,         nan],
                 [    -1.4211,      1.7445,      3.1249,      1.9389,      1.9842, 2.0633,      2.7237,     -2.8416,      3.0230,     -2.7041, 3.1346,         nan],
                 [    -1.5244,      1.7825,     -3.1349,      1.9350,      2.0589, 2.1902,      2.6275,     -1.1020,      2.9016,         nan, nan,         nan],
                 [    -1.1266,      0.6216,      3.0182,      1.9352,      2.0330, 2.0597,      3.0218,     -1.1527,     -2.3676,     -2.7324, nan,         nan],
                 [    -1.6891,      2.0191,     -3.0094,      1.9468,      2.1131, 2.2471,      2.4457,     -2.8059,      0.5180,         nan, nan,         nan],
                 [    -1.0525,     -0.1013,     -3.0147,      1.9677,      2.0847, 2.1087,      3.1402,      0.2295,     -0.2786,         nan, nan,         nan],
                 [    -2.1855,      0.5146,      2.8965,      1.9401,      2.2266, 2.1073,      1.9581,     -1.4947,      3.1090,     -0.0891, nan,         nan],
                 [     2.7184,      2.1836,      1.3244,      1.9390,      2.0611, 2.7559,      0.5810,     -1.4293,     -2.7450,      2.6851, -3.1292,      0.0016],
                 [     1.8037,      1.9978,      2.9519,      1.9412,      2.0317, 2.2638,     -0.3295,     -1.0505,         nan,         nan, nan,         nan],
                 [    -0.9451,      1.4802,      2.9436,      1.9330,      2.1320, 2.2217,     -3.0770,     -0.9302,     -3.0305,         nan, nan,         nan],
                 [    -1.2483,      1.9580,      3.0613,      1.9153,      2.0774, 2.4241,      2.8978,     -3.0975,     -0.9542,         nan, nan,         nan],
                 [    -1.2112,      1.3788,      2.7828,      1.9364,      2.1469, 1.9852,      2.9383,     -0.9342,      1.5948,         nan, nan,         nan],
                 [    -1.5769,     -0.6091,         nan,      1.9373,         nan, nan,      2.5730,     -1.1013,     -1.4846,         nan, nan,         nan]],
                 dtype=torch.float64)
    coords = torch.tensor([[[ -6.9940,  -2.6410,  -4.7930],
         [ -7.0450,  -1.3420,  -5.4590],
         [ -5.6540,  -0.7240,  -5.5630],
         [ -4.9150,  -0.6750,  -4.5770],
         [ -7.9890,  -0.4000,  -4.7150],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan]],

        [[ -4.9020,  -0.8370,  -6.5430],
         [ -3.5260,  -0.4000,  -6.7480],
         [ -3.4670,   1.0930,  -7.0500],
         [ -4.3540,   1.6320,  -7.7130],
         [ -2.8780,  -1.1860,  -7.8880],
         [ -1.1360,  -0.7860,  -8.1490],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan]],

        [[ -2.5760,   1.8310,  -6.2020],
         [ -2.3890,   3.2560,  -6.4560],
         [ -1.3770,   3.4870,  -7.5750],
         [ -0.8950,   2.5330,  -8.1900],
         [ -1.9370,   3.9730,  -5.1820],
         [ -0.6110,   3.4590,  -4.6510],
         [  0.4290,   3.6550,  -5.3150],
         [ -0.6070,   2.8500,  -3.5590],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan]],

        [[ -1.2470,   4.6590,  -8.0370],
         [ -0.3820,   5.1280,  -9.1150],
         [  1.0690,   4.7210,  -8.8750],
         [  1.8580,   4.6290,  -9.8170],
         [ -0.4830,   6.6490,  -9.2630],
         [ -1.8280,   7.1270,  -9.7930],
         [ -1.8730,   8.6240, -10.0510],
         [ -1.5740,   9.0540, -11.1880],
         [ -2.2090,   9.3750,  -9.1070],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan]],

        [[  1.3580,   4.2260,  -7.6730],
         [  2.7410,   3.9160,  -7.3290],
         [  2.9260,   2.4170,  -7.1260],
         [  4.0420,   1.9510,  -6.8890],
         [  3.1640,   4.6730,  -6.0660],
         [  3.1640,   6.1700,  -6.2270],
         [  4.2910,   6.8310,  -6.7010],
         [  4.2940,   8.2160,  -6.8500],
         [  3.1630,   8.9480,  -6.5230],
         [  2.0340,   8.2990,  -6.0500],
         [  2.0380,   6.9150,  -5.9030],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan]],

        [[  1.8520,   1.5790,  -7.4680],
         [  1.9180,   0.1270,  -7.4300],
         [  1.7300,  -0.4430,  -6.0360],
         [  2.1400,  -1.5720,  -5.7610],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan]],

        [[  1.2160,   0.3830,  -5.0980],
         [  0.9230,  -0.1190,  -3.7610],
         [ -0.4640,  -0.7510,  -3.7020],
         [ -1.3990,  -0.2690,  -4.3460],
         [  1.0300,   1.0070,  -2.7300],
         [  2.3880,   1.6320,  -2.6640],
         [  2.6070,   2.9650,  -2.9340],
         [  3.8950,   3.2300,  -2.7940],
         [  4.5170,   2.1170,  -2.4430],
         [  3.5950,   1.1030,  -2.3550],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan]],

        [[ -0.5370,  -1.9080,  -3.1280],
         [ -1.8420,  -2.5440,  -2.9830],
         [ -2.4040,  -2.2580,  -1.5920],
         [ -1.7500,  -2.5340,  -0.5830],
         [ -1.7570,  -4.0680,  -3.2230],
         [ -1.1920,  -4.3580,  -4.6190],
         [ -0.8260,  -5.8180,  -4.8470],
         [ -3.1310,  -4.7210,  -3.0430],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan]],

        [[ -3.4500,  -1.4880,  -1.5700],
         [ -4.1280,  -1.2110,  -0.3060],
         [ -5.0710,  -2.3490,   0.0720],
         [ -5.9070,  -2.7640,  -0.7340],
         [ -4.9010,   0.1060,  -0.3890],
         [ -5.2890,   0.6820,   0.9650],
         [ -5.9060,   2.0680,   0.8270],
         [ -5.7350,   2.8860,   2.1000],
         [ -6.3300,   4.2490,   1.9660],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan]],

        [[ -4.5680,  -3.1800,   0.9710],
         [ -5.3960,  -4.2940,   1.4200],
         [ -6.2790,  -3.8770,   2.5920],
         [ -5.8270,  -3.1700,   3.4950],
         [ -4.5210,  -5.4840,   1.8240],
         [ -3.6750,  -6.1110,   0.7140],
         [ -2.6220,  -7.0380,   1.3100],
         [ -4.5610,  -6.8630,  -0.2730],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan]],

        [[ -7.5400,  -3.6210,   2.3500],
         [ -8.5260,  -3.2030,   3.3430],
         [ -8.7570,  -4.3010,   4.3750],
         [ -9.8960,  -4.5580,   4.7700],
         [ -9.8470,  -2.8340,   2.6670],
         [ -9.7580,  -1.6040,   1.7770],
         [-11.1730,  -0.4600,   2.0160],
         [-11.1530,   0.4090,   0.4230],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan]],

        [[ -7.7780,  -4.9720,   4.6890],
         [ -7.9230,  -5.9250,   5.7850],
         [ -7.4300,  -5.3400,   7.1050],
         [ -6.3180,  -4.8140,   7.1790],
         [ -7.1780,  -7.2240,   5.4690],
         [ -7.6550,  -8.3900,   6.3120],
         [ -8.8190,  -8.4430,   6.7190],
         [ -6.7600,  -9.3340,   6.5800],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan]],

        [[ -8.2630,  -5.1360,   8.1170],
         [ -8.0110,  -4.4540,   9.3890],
         [ -6.9040,  -5.1170,  10.2050],
         [ -6.2970,  -4.4740,  11.0660],
         [ -9.3560,  -4.5510,  10.1130],
         [-10.1400,  -5.5540,   9.3310],
         [ -9.4780,  -5.7410,   7.9960],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan]],

        [[ -6.3850,  -6.2370,   9.7890],
         [ -5.4410,  -6.9770,  10.6190],
         [ -4.1390,  -7.2430,   9.8690],
         [ -3.1490,  -7.6720,  10.4650],
         [ -6.0560,  -8.2960,  11.0870],
         [ -6.8910,  -8.1700,  12.3550],
         [ -7.5260,  -9.4830,  12.7720],
         [ -7.4260, -10.4900,  12.0640],
         [ -8.1850,  -9.4830,  13.9260],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan]],

        [[ -3.5920,  -6.5070,   8.9230],
         [ -2.2080,  -6.7240,   8.5160],
         [ -1.9710,  -6.2320,   7.0920],
         [ -2.7850,  -6.4810,   6.2010],
         [ -1.8400,  -8.2050,   8.6260],
         [ -1.4600,  -8.6430,  10.0310],
         [ -0.5980,  -9.8980,  10.0180],
         [  0.2520,  -9.9810,  11.2020],
         [  1.1160, -10.9620,  11.4480],
         [  1.2630, -11.9660,  10.5920],
         [  1.8400, -10.9380,  12.5580],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan]],

        [[ -1.0060,  -5.3670,   6.8900],
         [  0.3600,  -4.8770,   6.7310],
         [  0.6880,  -4.6230,   5.2640],
         [  0.3650,  -5.4400,   4.3990],
         [  1.3590,  -5.8740,   7.3200],
         [  1.1190,  -6.0690,   8.7030],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan]],

        [[  0.9040,  -3.4180,   4.9350],
         [  1.4400,  -2.8730,   3.6930],
         [  2.7800,  -3.5190,   3.3530],
         [  3.6740,  -3.5900,   4.1990],
         [  1.6110,  -1.3450,   3.7820],
         [  0.3720,  -0.7540,   4.1900],
         [  2.0220,  -0.7580,   2.4350],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan]],

        [[  2.8600,  -4.6290,   2.7250],
         [  4.0550,  -5.2320,   2.1440],
         [  4.4810,  -4.4440,   0.9070],
         [  3.6480,  -4.1030,   0.0630],
         [  3.8230,  -6.7150,   1.7780],
         [  5.1030,  -7.3430,   1.2300],
         [  3.3220,  -7.4910,   2.9940],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan]],

        [[  5.7000,  -3.7030,   0.9440],
         [  6.5290,  -2.9050,   0.0470],
         [  7.1900,  -3.7830,  -1.0100],
         [  7.8780,  -4.7520,  -0.6800],
         [  7.5980,  -2.1440,   0.8360],
         [  7.0500,  -1.3150,   1.9600],
         [  6.8780,  -1.7010,   3.2600],
         [  6.3470,  -0.6670,   3.9940],
         [  6.1660,   0.4130,   3.1720],
         [  5.6600,   1.6840,   3.4520],
         [  5.5920,   2.5810,   2.4170],
         [  6.0140,   2.2410,   1.1220],
         [  6.5180,   0.9770,   0.8410],
         [  6.5990,   0.0400,   1.8800],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan]],

        [[  6.5540,  -4.2200,  -2.0840],
         [  7.4420,  -4.6210,  -3.1700],
         [  7.7320,  -3.4490,  -4.0990],
         [  6.8660,  -2.6020,  -4.3300],
         [  6.8290,  -5.7780,  -3.9650],
         [  6.6380,  -7.0380,  -3.1550],
         [  7.6770,  -7.9520,  -2.9990],
         [  7.5060,  -9.1150,  -2.2550],
         [  6.2840,  -9.3680,  -1.6600],
         [  6.1090, -10.5170,  -0.9230],
         [  5.2370,  -8.4760,  -1.8000],
         [  5.4200,  -7.3160,  -2.5460],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan],
         [     nan,      nan,      nan]]], dtype=torch.float64)
    # yapf: enable
    p = SCNProtein(
        coordinates=coords,
        angles=angles,
        sequence="ACDEFGHIKLMNPQRSTVWY",
        mask="+"*len("ACDEFGHIKLMNPQRSTVWY"),
        id="ALFA_1_A",
        split="train")
    p.torch()
    return p


@pytest.fixture
def palpha():
    return alphabet_protein()

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
    p = alphabet_protein()
    # p.to_pdb("./alphabet_build01.pdb")
    # p.fastbuild(inplace=True)
    # p.to_pdb("./alphabet_build02.pdb")
    p.fastbuild(add_hydrogens=True, inplace=True)
    p.to_pdb("/home/jok120/Downloads/build_alfa04.pdb")
