from sidechainnet.structure import trig_transform
from torch.nn.functional import mse_loss
import numpy as np
import torch

from pytest import approx

from sidechainnet.examples.losses import angle_diff

# There are two ways of thinking about angle loss. 1) Value agnostic, 2) Value-aware
# The simplest way to handle this would be to treat the loss function as an
# agnostic tool that only needs to know the appropriate mask. It the zeros out
# (via selection) irrelevant values before computing a raw MSE.

# For clarity, lets say there are 3 angles per residue. They may either be represented
# in sin/cos terms (d=6) or radians (d=3). We'll also set the length to a simple 5.


def test_mse_radians():
    # 1. Radian values
    # 1a. No missing values
    true = torch.tensor([[
        [1.8, 2.4, 3.1],
        [2.1, 0.2, 3.0],
        [0.5, 0.1, 0.3],
        [1.6, 1.1, 0.9],
        [2.4, 2.8, 0.6],
    ]])
    pred = torch.tensor([[
        [1.8, 2.4, 3.1],
        [2.1, 0.2, 3.0],
        [0.5, 0.1, 0.3],
        [1.6, 1.1, 0.9],
        [2.4, 2.8, 0.6],
    ]])
    mask = ~true.isnan()
    result = mse_loss(pred[mask], true[mask])
    assert result == 0

    # 1b. Missing first and last residues
    true = torch.tensor([[
        [np.nan, np.nan, np.nan],
        [2.1, 0.2, 3.0],
        [0.5, 0.1, 0.3],
        [1.6, 1.1, 0.9],
        [np.nan, np.nan, np.nan],
    ]])
    pred = torch.tensor([[
        [1.8, 2.4, 3.1],
        [2.1, 0.2, 3.0],
        [0.5, 0.1, 0.3],
        [1.6, 1.1, 0.9],
        [2.4, 2.8, 0.6],
    ]])
    mask = ~true.isnan()
    result = mse_loss(pred[mask], true[mask])
    assert result == 0

    # 1b. Missing angles in the middle
    true = torch.tensor([[
        [np.nan, np.nan, np.nan],
        [np.nan, 0.2, 3.0],
        [0.5, np.nan, 0.3],
        [1.6, 1.1, 0.9],
        [np.nan, np.nan, 0.6],
    ]])
    pred = torch.tensor([[
        [1.8, 2.4, 3.1],
        [2.1, 0.2, 3.0],
        [0.5, 0.1, 0.3],
        [1.6, 1.1, 0.9],
        [2.4, 2.8, 0.6],
    ]])
    mask = ~true.isnan()
    result = mse_loss(pred[mask], true[mask])
    assert result == 0


def test_angle_ae():

    def T(n):
        return torch.tensor([n])

    pi = np.pi

    # Test that angle diff understands signed differences
    assert angle_diff(T(-pi / 2), T(pi / 2)) == approx(-pi)
    assert angle_diff(T(pi / 2), T(-pi / 2)) == approx(pi)

    assert angle_diff(T(11 * pi / 6), T(7 * pi / 6)) == approx(4 * pi / 6)
    assert angle_diff(T(7 * pi / 6), T(11 * pi / 6)) == approx(-4 * pi / 6)

    # Test that full 2pi rotations don't matter for 2nd argument
    assert angle_diff(T(.1), T(.2)) == approx(T(-.1))
    assert angle_diff(T(.1), T(.2 + np.pi * 2)) == approx(T(-.1), rel=1e-5)
    assert angle_diff(T(.1), T(.2 - np.pi * 2)) == approx(T(-.1), rel=1e-5)

    assert angle_diff(T(.2), T(.1)) == approx(T(.1))
    assert angle_diff(T(.2), T(.1 - np.pi * 2)) == approx(T(.1), rel=1e-5)
    assert angle_diff(T(.2), T(.1 + np.pi * 2)) == approx(T(.1), rel=1e-5)

    # Test that full 2pi rotations don't matter for 1st argument
    assert angle_diff(T(.1 + np.pi * 2), T(.2)) == approx(T(-.1), rel=1e-5)
    assert angle_diff(T(.1 - np.pi * 2), T(.2)) == approx(T(-.1), rel=1e-5)
    assert angle_diff(T(.2 + np.pi * 2), T(.1)) == approx(T(.1), rel=1e-5)
    assert angle_diff(T(.2 - np.pi * 2), T(.1)) == approx(T(.1), rel=1e-5)


def test_mse_trig():
    # 1. Radian values
    # 1a. No missing values
    true = torch.tensor([[
        [1.8, 2.4, 3.1],
        [2.1, 0.2, 3.0],
        [0.5, 0.1, 0.3],
        [1.6, 1.1, 0.9],
        [2.4, 2.8, 0.6],
    ]])
    pred = torch.tensor([[
        [1.8, 2.4, 3.1],
        [2.1, 0.2, 3.0],
        [0.5, 0.1, 0.3],
        [1.6, 1.1, 0.9],
        [2.4, 2.8, 0.6],
    ]])
    true = trig_transform(true)
    pred = trig_transform(pred)
    mask = ~true.isnan()
    result = mse_loss(pred[mask], true[mask])
    assert result == 0

    # 1b. Missing first and last residues
    true = torch.tensor([[
        [np.nan, np.nan, np.nan],
        [2.1, 0.2, 3.0],
        [0.5, 0.1, 0.3],
        [1.6, 1.1, 0.9],
        [np.nan, np.nan, np.nan],
    ]])
    pred = torch.tensor([[
        [1.8, 2.4, 3.1],
        [2.1, 0.2, 3.0],
        [0.5, 0.1, 0.3],
        [1.6, 1.1, 0.9],
        [2.4, 2.8, 0.6],
    ]])
    true = trig_transform(true)
    pred = trig_transform(pred)
    mask = ~true.isnan()
    result = mse_loss(pred[mask], true[mask])
    assert result == 0

    # 1c. Missing angles in the middle
    true = torch.tensor([[
        [np.nan, np.nan, np.nan],
        [np.nan, 0.2, 3.0],
        [0.5, np.nan, 0.3],
        [1.6, 1.1, 0.9],
        [np.nan, np.nan, 0.6],
    ]])
    pred = torch.tensor([[
        [1.8, 2.4, 3.1],
        [2.1, 0.2, 3.0],
        [0.5, 0.1, 0.3],
        [1.6, 1.1, 0.9],
        [2.4, 2.8, 0.6],
    ]])
    true = trig_transform(true)
    pred = trig_transform(pred)
    mask = ~true.isnan()
    result = mse_loss(pred[mask], true[mask])
    assert result == 0

    # 1d. Flatten the last dimension (instead of by two)
    true = torch.tensor([[
        [np.nan, np.nan, np.nan],
        [np.nan, 0.2, 3.0],
        [0.5, np.nan, 0.3],
        [1.6, 1.1, 0.9],
        [np.nan, np.nan, 0.6],
    ]])
    pred = torch.tensor([[
        [1.8, 2.4, 3.1],
        [2.1, 0.2, 3.0],
        [0.5, 0.1, 0.3],
        [1.6, 1.1, 0.9],
        [2.4, 2.8, 0.6],
    ]])
    true = trig_transform(true).flatten(start_dim=-2)
    pred = trig_transform(pred).flatten(start_dim=-2)
    mask = ~true.isnan()
    result = mse_loss(pred[mask], true[mask])
    assert result == 0