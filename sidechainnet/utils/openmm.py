"""Torch layer for forward and backward passes of energy computation.

Force is the negative gradient of potential energy, i.e.  F(x) = −∇U(x). Luckily,
this is done for us by OpenMM and we simply must return the computed forces rather
than perform any differentiation directly.
"""

import torch


class OpenMMEnergy(torch.autograd.Function):
    """A force-field based loss function."""

    @staticmethod
    def forward(ctx, protein, coords, output_rep="heavy", sum_hydrogens=False):
        """Compute potential energy of the protein system and save atomic forces."""
        # Update atomic position for all atoms when optimizing hydrogens directly
        protein.update_hydrogens(coords)

        # Compute potential energy
        energy = torch.tensor(protein.get_energy()._value, requires_grad=True, dtype=torch.float64)

        # Save context
        forces, raw_forces = protein.get_forces(output_rep, sum_hydrogens)
        ctx.forces = forces
        protein._force_array = forces
        protein._force_array_raw = raw_forces

        return energy

    @staticmethod
    def backward(ctx, grad_output=None):
        """Return the negative force acting on each heavy atom."""
        # Unpack saved variables
        forces = ctx.forces

        return None, -forces * grad_output, None, None
