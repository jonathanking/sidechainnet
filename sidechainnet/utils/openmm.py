"""Torch layer for forward and backward passes of energy computation.

Force is the negative gradient of potential energy, i.e.  F(x) = −∇U(x). Luckily,
this is done for us by OpenMM and we simply must return the computed forces rather
than perform any differentiation directly.
"""

import torch
from torch.autograd.functional import jacobian as get_jacobian


class OpenMMEnergy(torch.autograd.Function):
    """A force-field based loss function."""

    @staticmethod
    def forward(ctx, protein, coords, output_rep="all", jacobian=None):
        """Compute potential energy of the protein system and save atomic forces."""
        # Compute Jacobian if not provided
        if jacobian is not None:
            protein.add_hydrogens(coords=coords)
            ctx.dxh_dx = jacobian
        else:
            new_coords = coords.detach().clone().requires_grad_(True)
            add_h = _get_alias(protein)
            ctx.dxh_dx = get_jacobian(add_h, new_coords)

        # Compute potential energy and forces
        energy = torch.tensor(protein.get_energy()._value,
                              requires_grad=True,
                              dtype=torch.float64)
        forces = protein.get_forces(output_rep)

        # Save context
        ctx.forces = forces
        protein._forces = forces
        protein._dxh_dx = ctx.dxh_dx

        return energy

    @staticmethod
    def backward(ctx, grad_output=None):
        """Return the negative force acting on each heavy atom."""
        forces = ctx.forces.view(1, -1)                    # (1 x M)
        dxh_dx = ctx.dxh_dx                                # (26L x 3 x 14L x 3)
        a, b, c, d = dxh_dx.shape
        dxh_dx = dxh_dx.view(a * b, c * d)                 # (M x N)

        de_dx = torch.matmul(-forces, dxh_dx).view(-1, 3)  # (1 x N) -> (14L x 3)

        return None, de_dx * grad_output, None, None, None


class OpenMMEnergyH(torch.autograd.Function):
    """A force-field based loss function."""

    @staticmethod
    def forward(ctx, protein, hcoords, force_scaling=1):
        """Compute potential energy of the protein system and save atomic forces."""
        # Update protein's hcoords
        protein.update_hydrogens(hcoords / 10)

        # Compute potential energy and forces
        energy = torch.tensor(protein.get_energy()._value,
                              requires_grad=True,
                              dtype=torch.float64)
        forces = protein.get_forces() / 10

        # Save context
        ctx.forces = forces
        protein._forces = forces
        ctx.gradient_scale = force_scaling

        return energy

    @staticmethod
    def backward(ctx, grad_output=None):
        """Return the negative force acting on each atom."""
        forces = ctx.forces

        return None, torch.tensor(-forces) * grad_output * ctx.gradient_scale, None


def _get_alias(protein):

    def add_h(tcoords):
        """Mimics f(X) -> X_H."""
        protein.add_hydrogens(coords=tcoords)
        return protein.hcoords

    return add_h
