"""Torch layer for forward and backward passes of energy computation.

Force is the negative gradient of potential energy, i.e.  F(x) = −∇U(x). Luckily,
this is done for us by OpenMM and we simply must return the computed forces rather
than perform any differentiation directly.
"""

import torch


class OpenMMEnergy(torch.autograd.Function):
    """A force-field based loss function."""

    @staticmethod
    def forward(ctx, protein, coords):
        """Compute potential energy of the protein system and save atomic forces."""
        # Add hydrogens to provided coordinates, update "positions" for OpenMM
        protein.add_hydrogens(coords=coords)

        # Compute potential energy
        energy = torch.tensor(protein.get_energy()._value, requires_grad=True, dtype=torch.float64)

        # Save context
        ctx.forces = protein.get_forces()
        # print("Forces:", ctx.forces)

        return energy

    @staticmethod
    def backward(ctx, grad_output=None):
        """Return the negative force acting on each heavy atom."""
        # Unpack saved variables
        forces = ctx.forces

        return None, -forces
