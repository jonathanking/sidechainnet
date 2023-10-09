"""This module contains the necessary loss functions to train OpenFold with OpenMMLoss."""

import argparse
import torch
from typing import Dict, List
import numpy as np

from openfold.np.protein import Protein
from sidechainnet.dataloaders.SCNProtein import SCNProtein
from sidechainnet.structure.build_info import ATOM_MAP_HEAVY, NUM_COORDS_PER_RES
from sidechainnet.utils.openmm_loss import OpenMMEnergyH, OpenMMEnergy
from openfold.utils.feats import atom37_to_atom14

import openfold

_weight = argparse.Namespace()
_weight.weight = 1.0


def openmm_loss(
        model_output: Dict,  # Complete model output
        model_input: Dict,  # Complete model input
        force_scaling=None,
        force_clipping_val=None,
        scale_by_length=False,
        squash=False,
        squash_factor=100.0,
        modified_sigmoid=False,
        modified_sigmoid_params=(1,1,1,1),
        add_relu=False,
) -> torch.Tensor:
    """Return a loss value based on the OpenMM energies of the predicted structures.
    
    A wrapper around SidechainNet's OpenMMEnergyH loss function for use in OpenFold.

    Args:
        model_output (Dict): The model output.
        model_input (Dict): The model input.

    Returns:
        torch.Tensor: The energy of the predicted structure as measured by OpenMM.
    """
    assert model_output["final_atom_positions"].shape[-1] == 3
    assert model_output["final_atom_positions"].shape[-2] == 37

    # Convert the model output into a list of SidechainNet proteins
    # scn_proteins = _create_scn_proteins_from_openfold_output(model_input, model_output)
    scn_proteins = [p for p in iterate_over_model_input_output(model_input, model_output) if p is not None]
    scn_proteins_gt = [p for p in iterate_over_model_input_output(model_input, model_output, ground_truth=True) if p is not None]
    scn_proteins_no_hy = [p.copy() for p in scn_proteins]

    # Add hydrogens to each protein
    for protein in scn_proteins:
        #TODO-JK: What method should we use to add hydrogens?
        protein.add_hydrogens(add_to_heavy_atoms=True)

    loss = OpenMMEnergyH()

    # Now, compute the energy of each protein
    total_energy = torch.tensor(0.0, device=model_output["final_atom_positions"].device)
    total_energy_raw = torch.tensor(0.0, device=model_output["final_atom_positions"].device)
    for protein in scn_proteins:
        if "X" in protein.seq:
            # Skip proteins with unknown residues, they cannot have their energy computed
            continue
        protein_energy = loss.apply(protein,
                                    protein.hcoords,
                                    force_scaling,
                                    force_clipping_val)
        protein_energy_raw = protein_energy.clone()
        total_energy_raw += protein_energy_raw
        if scale_by_length:
            protein_energy /= len(protein)
        if add_relu:
            relu_component = protein_energy * 10**-12  # if protein_energy > 0 else 0
        if squash and protein_energy > 0:
            protein_energy = protein_energy * (squash_factor/(squash_factor+protein_energy))
        elif modified_sigmoid:
            a, b, c, d = modified_sigmoid_params
            protein_energy = (1/a + torch.exp(-(d*protein_energy+b)/c))**(-1) - (a-1)
        if add_relu:
            protein_energy += relu_component

        total_energy += protein_energy

    return total_energy, scn_proteins_no_hy, scn_proteins_gt, total_energy_raw


def iterate_over_model_input_output(model_input, model_output, ground_truth=False):
    """Iterate over the model input and output, yielding the input and output for each protein in the batch.
    
    Args:
        model_input (Dict): The model input.
        model_output (Dict): The model output.
    
    Yields:
        protein (SCNProtein): SidechainNet protein created from model input/output.
    """
    batch_size = model_input["aatype"].shape[0]

    if ground_truth:
        batch_coords14 = model_input["all_atom_positions"]
        batch_coords14 = atom37_to_atom14(batch_coords14, model_input, no_batch_dims=2)
    else:
        batch_coords14 = model_output["sm"]["positions"][-1]  

    for i in range(batch_size):
        # Create sequence variable from model input
        aatype = model_input["aatype"][i]
        seq_length = model_input["seq_length"][i]
        aatype = aatype[:seq_length]
        seq = convert_openfold_aatype_to_str_seq(aatype)

        # Create sequence mask variable from model input
        # seq_mask = model_input["seq_mask"][i]  # Should be str of +/-
        # seq_mask = seq_mask[:seq_length]
        # seq_mask = convert_openfold_seq_mask_to_str_seq_mask(seq_mask)
        # Because we are using a prediction that has no missing pieces, the mask should be all +
        seq_mask = "".join(["+" if char == 1 else "-" for char in model_input["seq_mask"][i, :seq_length]])

        # Create atom14 variable from model output
        coords14 = batch_coords14[i]
        coords14 = coords14[:seq_length]
        coords15 = convert_openfold_atom14_to_scn_atom15(coords14, seq)

        protein = SCNProtein(coordinates=coords15, sequence=seq, mask=seq_mask, id=model_input['name'][i])

        if "X" in protein.seq:
            # Skip proteins with unknown residues, they cannot have their energy computed
            yield None
        else:
            yield protein


def convert_openfold_aatype_to_str_seq(aatype):
    """Convert a tensor containing OpenFold's aatype ints to a string sequence."""
    return openfold.np.residue_constants.aatype_to_str_sequence(aatype)

def convert_openfold_seq_mask_to_str_seq_mask(seq_mask):
    """Convert a mask containing 1s and 0s to a string mask containing + and -."""
    str_mask = ''.join(['+' if x == 1 else '-' for x in seq_mask])
    return str_mask


def convert_openfold_atom14_to_scn_atom15(atom14, seq):
    # TODO-JK make sure that atom14 is in the correct order, some atoms may be out of order
    # TODO-JK this could be sped up by using aatype integers for selection instead of a dict
    new_atoms = torch.zeros(
        atom14.shape[0], NUM_COORDS_PER_RES, 3, device=atom14.device,
        dtype=atom14.dtype) * torch.nan
    # mapping = get_openfold_to_sidechainnet_mapping()
    for i, res in enumerate(seq):
        source_selection, target_selection = OPENFOLD_TO_SCN_MAP[res]
        new_atoms[i][target_selection] = atom14[i][source_selection]
    return new_atoms


def get_openfold_to_sidechainnet_mapping():
    from sidechainnet.structure.build_info import ATOM_MAP_HEAVY
    from sidechainnet.utils.sequence import ONE_TO_THREE_LETTER_MAP, THREE_TO_ONE_LETTER_MAP
    from openfold.np.residue_constants import restype_name_to_atom14_names

    openfold_idx_to_scn_idx = {}
    openfold_name_to_openfold_idx = {}
    for resname, atom_names in restype_name_to_atom14_names.items():
        openfold_name_to_openfold_idx[resname] = {
            atom_name: i
            for i, atom_name in enumerate(atom_names) if atom_name
        }

    scn_name_to_scn_idx = {}
    for resid, atom_names in ATOM_MAP_HEAVY.items():
        resname = ONE_TO_THREE_LETTER_MAP[resid]
        scn_name_to_scn_idx[resname] = {
            atom_name: i
            for i, atom_name in enumerate(atom_names) if atom_name != "PAD"
        }

    openfold_idx_to_scn_idx = {}
    # Now, compare the two dictionaries to make note of where the differences are
    for (resname, scn_atomnames) in scn_name_to_scn_idx.items():
        rescode = THREE_TO_ONE_LETTER_MAP[resname]
        openfold_idx_to_scn_idx[rescode] = {}
        for scn_atomname, scn_idx in scn_atomnames.items():
            try:
                openfold_idx = openfold_name_to_openfold_idx[resname][scn_atomname]
            except KeyError:
                # openfold_idx = scn_atomname
                # don't add this atom to the dictionary
                continue
            openfold_idx_to_scn_idx[rescode][openfold_idx] = scn_idx
        openfold_idx_to_scn_idx[rescode] = (
            torch.tensor(list(openfold_idx_to_scn_idx[rescode].keys())),
            torch.tensor(list(openfold_idx_to_scn_idx[rescode].values())))

    # For unknown residues (X), interpret as Glycine
    openfold_idx_to_scn_idx["X"] = openfold_idx_to_scn_idx["G"]
    

    return openfold_idx_to_scn_idx


class OpenMMLR:
    """Learning rate schedule that starts at 1e-4 and increases to 1 over 1000 steps"""
    def __init__(self, start=1e-4, end=1, steps=1000, cur_step=0):
        self.lr = np.linspace(start, end, steps)
        self.steps = steps
        self.cur_step = cur_step
    
    def step(self, new_step=None):
        if new_step is not None:
            self.cur_step = new_step
        else:
            self.cur_step += 1

    def get_lr(self, step=None):
        if step is None:
            step = self.cur_step
        if self.cur_step >= self.steps:
            return self.lr[-1]
        else:
            return self.lr[self.cur_step]

    def get_lr_and_step(self):
        val = self.get_lr()
        self.step()
        return val


OPENFOLD_TO_SCN_MAP = get_openfold_to_sidechainnet_mapping()


# def _create_ith_scn_protein_from_openfold_output(
#         i: int,
#         model_input: Dict,  # this represents the batch
#         model_output: Dict,  # this represents the model's predictions
# ) -> SCNProtein:
#     """Take a batch of predictions and convert the ith prediction into a SCNProtein.

#     Args:
#         i (int): index of the protein in the batch.
#         input (Dict): model input.
#         output (Dict): model output.

#     Returns:
#         SCNProtein: parsed SidechainNet protein.
#     """
#     pass


# def _create_scn_proteins_from_openfold_output(model_input: Dict,
#                                               model_output: Dict) -> List[SCNProtein]:
#     """Take a batch of predictions and convert them into a list of SidechainNet proteins.

#     Args:
#         input (Dict): model input.
#         output (Dict): model output.

#     Returns:
#         List[SCNProtein]: List of parsed SidechainNet proteins.
#     """
#     # TODO-JK : this likely won't work, since I can't iterate over the batch input/output
#     print(model_input.keys())
#     print(model_output.keys())
#     proteins = [
#         _create_ith_scn_protein_from_openfold_output(i, _in, _out)
#         for i, (_in, _out) in enumerate(zip(model_input, model_output))
#     ]
#     return proteins