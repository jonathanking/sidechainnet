"""Torch layer for forward and backward passes of energy computation.

Force is the negative gradient of potential energy, i.e.  F(x) = −∇U(x). Luckily,
this is done for us by OpenMM and we simply must return the computed forces rather
than perform any differentiation directly.
"""

import torch
from torch.autograd.functional import jacobian as get_jacobian


class OpenMMEnergyH(torch.autograd.Function):
    """A force-field based loss function."""

    @staticmethod
    def forward(ctx, protein, hcoords, force_scaling=None, force_clipping_val=None):
        """Compute protein potential energy and save atomic forces for the backwards pass.

        Args:
            ctx (torch.autograd.Function): The autograd context.
            protein (sidechainnet.SCNProtein): A sidechainnet protein.
            hcoords (torch.Tensor): The current all-atom coordinates, including hydrogens.
            force_scaling (float): A factor by which to scale the forces.
            force_clipping_val (float): A value by which to clip the forces.
        
        Returns:
            torch.Tensor: The potential energy of the protein.
        """
        # Set reasonable defaults
        if force_scaling is None:
            force_scaling = 1
        if force_clipping_val is None:
            force_clipping_val = 1e6

        # Update protein's hcoords, scaled behind the scenes to match OpenMM's units
        protein.update_hydrogens_for_openmm(hcoords)

        # Compute potential energy and forces
        energy = torch.tensor(protein.get_energy(return_unitless_kjmol=True),
                              requires_grad=True,
                              dtype=torch.float64,
                              device=protein.device)
        forces = protein.get_forces()

        # Some forces may have inf/-inf, so we clip large values to avoid nan gradients
        forces[forces > force_clipping_val] = force_clipping_val
        forces[forces < -force_clipping_val] = -force_clipping_val

        # Save context
        ctx.forces = forces
        ctx.grad_scale = force_scaling
        ctx.device = protein.device
        ctx.coord_shape = hcoords.shape

        return energy

    @staticmethod
    def backward(ctx, grad_output=None):
        """Return the negative force acting on each atom."""
        return None, torch.tensor(
            -ctx.forces.reshape(*ctx.coord_shape),
            device=ctx.device) * grad_output * ctx.grad_scale, None, None


class OpenMMEnergy(torch.autograd.Function):
    """Not utilized in our research. A skeleton for OpenMM loss without hydrogens."""

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
        forces = ctx.forces.view(1, -1)  # (1 x M)
        dxh_dx = ctx.dxh_dx  # (26L x 3 x 14L x 3)
        a, b, c, d = dxh_dx.shape
        dxh_dx = dxh_dx.view(a * b, c * d)  # (M x N)

        de_dx = torch.matmul(-forces, dxh_dx).view(-1, 3)  # (1 x N) -> (14L x 3)

        return None, de_dx * grad_output, None, None, None



def _get_alias(protein):

    def add_h(tcoords):
        """Mimics f(X) -> X_H."""
        protein.add_hydrogens(coords=tcoords)
        return protein.hcoords

    return add_h
