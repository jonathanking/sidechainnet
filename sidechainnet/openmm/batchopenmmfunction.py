import torch
import numpy as np
from torch.autograd import Function
import sidechainnet as scn
from .openmmpdb import OpenMMPDB
from sidechainnet.utils.sequence import VOCAB
import random
import warnings
warnings.filterwarnings("ignore")


class OpenMMBatchFunction(Function):

    @staticmethod
    def forward(ctx, coords, input):
        assert input.shape[0] == coords.shape[0], "Batch Size inconsistency"
        pdb_mmlist = []
        for protein_index in range(input.shape[0]):
            seq = ''.join([
                VOCAB.int2char(seq_int.index(1))
                for seq_int in input[protein_index, :, :21].tolist()
            ]).replace('_', '')
            coord = coords[protein_index][:len(seq) * 14]
            sb = scn.StructureBuilder(seq, crd=coord)
            pdbstr = sb.to_pdbstr()
            pdb_mm = OpenMMPDB(pdbstr)
            pdb_mmlist.append(pdb_mm)
        ctx.pdb_mmlist = pdb_mmlist
        ctx.coords = coords
        ret_val = torch.as_tensor(
            [pdb_mm.get_potential_energy()._value for pdb_mm in pdb_mmlist])
        return ret_val

    @staticmethod
    def backward(ctx, grad_output=None):
        coords = ctx.coords
        pdb_mmlist = ctx.pdb_mmlist
        coords_sum = coords.sum(axis=2)
        force_arr = np.zeros((coords.shape[0], coords.shape[1], coords.shape[2]),
                             dtype=np.float64)
        for batch_index in range(coords.shape[0]):
            pdb_mm = pdb_mmlist[batch_index]
            coord_sum = coords_sum[batch_index]
            forces = [force._value for force in pdb_mm.get_forces_per_atoms().values()]
            assert len(forces) == coord_sum[
                coord_sum != 0].shape[0], "Not generating force for all existing atoms"
            for i in range(coords[batch_index].shape[0]):
                if coord_sum[i] != 0:
                    force = forces.pop(0)
                    force_arr[batch_index][i] = force / np.linalg.norm(force)
        return torch.tensor(force_arr, dtype=torch.float64), None


def inject_noise(coords):
    for index in range(coords.shape[0]):
        nonzero = np.nonzero(coords[index])
        for i in range(nonzero.shape[0]):
            print(coords[index, nonzero[i][0]])
            coords[index, nonzero[i][0], nonzero[i][1]] += random.randint(0, 1)
    return coords


def errorcheck(data, data1):
    assert data.shape == data1.shape, "Dimension Mismatch"
    data_non_zeros = np.nonzero(data).reshape(1, -1)
    data_non_zeros.squeeze()
    data1_non_zeros = np.nonzero(data1).reshape(1, -1)
    data1_non_zeros.squeeze()
    assert data_non_zeros.tolist() == data1_non_zeros.tolist(
    ), "Gradient for non existent atom"


if __name__ == '__main__':
    random.seed(10)
    torch.manual_seed(10)
    warnings.filterwarnings("ignore")
    openmmf = OpenMMBatchFunction.apply
    dataloaders = scn.load(casp_version=12,
                           with_pytorch="dataloaders",
                           batch_size=2,
                           num_workers=0,
                           use_dynamic_batch_size=False)
    for protein_id, model_input, true_angles, true_coords in dataloaders['train']:
        print(protein_id)
        coords = inject_noise(true_coords)
        coords.requires_grad = True
        for i in range(50):
            loss = openmmf(coords, model_input)
            loss.backward(torch.ones_like(loss))
            coords.data += 0.001 * coords.grad.data
            errorcheck(coords.data, coords.grad.data)
            coords.grad.data.zero_()
            print(loss)
        break
