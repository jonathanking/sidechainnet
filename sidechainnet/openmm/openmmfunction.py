import torch
import numpy as np
from torch.autograd import Function
import sidechainnet as scn
from .openmmpdb import OpenMMPDB


class OpenMMFunction(Function):

    @staticmethod
    def forward(ctx, coord, seq):
        sb = scn.StructureBuilder(seq, crd=coord)
        sb._initialize_coordinates_and_PdbCreator()
        pdbstr = sb.pdb_creator.get_pdb_string()
        pdb_mm = OpenMMPDB(pdbstr)
        ctx.pdb_mm = pdb_mm
        ctx.coord = coord
        ctx.save_for_backward(coord)
        ret_val = torch.as_tensor(pdb_mm.get_potential_energy()._value)
        return ret_val

    @staticmethod
    def backward(ctx, grad_output=None):
        coord = ctx.coord
        pdb_mm = ctx.pdb_mm
        forces = [force._value for force in pdb_mm.get_forces_per_atoms().values()]
        coord_sum = coord.sum(axis=1)
        # If coord_ten.sum(axis=1)[i] is 0, insert 0,0,0 in the arr, else insert one by one from forces
        force_arr = np.zeros((coord.shape[0], coord.shape[1]), dtype=np.float64)
        for i in range(force_arr.shape[0]):
            if coord_sum[i] != 0:
                force = forces.pop(0)
                force_arr[i] = force / np.linalg.norm(force)
        return torch.tensor(force_arr, dtype=torch.float64), None


if __name__ == '__main__':
    from torch.autograd import gradcheck
    #openmmf = OpenMMFunction.apply
    #data = scn.load(casp_version=12, thinning=30)
    # input = torch.tensor(data['train']['crd'][0], dtype=torch.float64, requires_grad=True), data['train']['seq'][0]
    # test = gradcheck(openmmf, input, eps=1e-6, atol=1e-4)
    # print(test)
