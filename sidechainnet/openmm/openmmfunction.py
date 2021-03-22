"""Torch layer for forward and backward passes of energy computation.

Force is the negative gradient of potential energy, i.e.  F(x) = −∇U(x). Luckily,
this is done for us by OpenMM and we simply must return the computed forces rather
than perform any differentiation directly.
"""

import torch
import numpy as np
from torch.autograd import Function
import sidechainnet as scn
from sidechainnet.openmm.openmmpdb import OpenMMPDB


class OpenMMFunction(Function):

    @staticmethod
    def forward(ctx, coord, seq):
        # Take coordinates and build a PDB file
        sb = scn.StructureBuilder(seq, crd=coord)
        sb._initialize_coordinates_and_PdbCreator()
        pdbstr = sb.pdb_creator.get_pdb_string()

        # Make OpenMMPDB object to compute P.E.
        pdb_mm = OpenMMPDB(pdbstr)

        # Save variables for backwards pass
        ctx.pdb_mm = pdb_mm
        ctx.coord = coord
        ctx.save_for_backward(coord)

        return torch.as_tensor(pdb_mm.get_potential_energy()._value)

    @staticmethod
    def backward(ctx, grad_output=None):
        # Unpack saved variables
        coord = ctx.coord
        pdb_mm = ctx.pdb_mm

        # Compute forces per atom for relevant atoms only
        forces = [force._value for force in pdb_mm.get_forces_per_atoms().values()]
        coord_sum = coord.sum(axis=1)

        # Fill in relevant forces, keeping irrelevant items 0
        force_arr = np.zeros_like(coord, dtype=np.float64)
        for i in range(force_arr.shape[0]):
            if coord_sum[i] != 0:
                force = forces.pop(0)
                force_arr[i] = force / np.linalg.norm(force)
        return torch.tensor(force_arr, dtype=torch.float64), None


if __name__ == '__main__':
    from torch.autograd import gradcheck
    openmmf = OpenMMFunction()
    data = scn.load("debug")
    _input = torch.tensor(data['train']['crd'][0], dtype=torch.float64, requires_grad=True), data['train']['seq'][0]
    output = openmmf.apply(*_input)
    grads = output.backward()
    test = gradcheck(OpenMMFunction.apply, _input, eps=1e-6, atol=1e-4)
    print(test)
