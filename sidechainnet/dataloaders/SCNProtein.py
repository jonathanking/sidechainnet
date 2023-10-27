"""A high-level interface for working with SidechainNet proteins in a dataset.

Available SCNProtein attributes include:
    * coords
    * angles
    * seq
    * unmodified_seq
    * mask
    * evolutionary
    * secondary_structure
    * resolution
    * is_modified
    * id
    * split

Other features:
    * add non-terminal hydrogens to a protein's coordinates with SCNProtein.add_hydrogens
    * visualize proteins with SCNProtein.to_3Dmol()
    * write PDB files for proteins with SCNProtein.to_PDB()
"""
import pickle
import random
import warnings

import numpy as np
import openmm
import prody
import sidechainnet as scn
from sidechainnet.utils.download import get_resolution_from_pdbid
import sidechainnet
import sidechainnet.structure.fastbuild as fastbuild
from sidechainnet.utils.openmm_loss import OpenMMEnergyH
import torch
from openmm import Platform
from openmm.app import Topology
from openmm.app import element as elem
from openmm.app.forcefield import ForceField, HBonds
from openmm.app.modeller import Modeller
from openmm.app.pdbfile import PDBFile
from openmm.openmm import LangevinMiddleIntegrator
from openmm.unit import kelvin, nanometer, picosecond, picoseconds, angstrom, mole, kilojoule
from sidechainnet.structure.build_info import (ATOM_MAP_H, ATOM_MAP_HEAVY,
                                               GLOBAL_PAD_CHAR, HEAVY_ATOM_MASK_TENSOR,
                                               NUM_COORDS_PER_RES,
                                               NUM_COORDS_PER_RES_W_HYDROGENS,
                                               SC_HBUILD_INFO)
from sidechainnet.structure.structure import coord_generator
from sidechainnet.utils.download import MAX_SEQ_LEN
from sidechainnet.utils.sequence import (ONE_TO_THREE_LETTER_MAP, VOCAB, DSSPVocabulary)
import simtk.unit as unit
OPENMM_FORCEFIELDS = ['amber14/protein.ff15ipq.xml', 'implicit/gbn.xml']
OPENMM_PLATFORM = "CPU"  # CUDA"  # CUDA or CPU
SEED = 1234


class SCNProtein(object):
    """Represent one protein in SidechainNet. Created programmatically by SCNDataset."""
    def __init__(self, **kwargs) -> None:
        """Create a SCNProtein from keyword arguments."""
        super().__init__()
        self.coords = kwargs['coordinates'] if 'coordinates' in kwargs else None
        self.angles = kwargs['angles'] if 'angles' in kwargs else None
        self.seq = kwargs['sequence'] if 'sequence' in kwargs else None
        self.unmodified_seq = kwargs[
            'unmodified_seq'] if 'unmodified_seq' in kwargs else None
        self.mask = kwargs['mask'] if 'mask' in kwargs else None
        self.evolutionary = kwargs['evolutionary'] if 'evolutionary' in kwargs else None
        self.secondary_structure = kwargs[
            'secondary_structure'] if 'secondary_structure' in kwargs else None
        self.resolution = kwargs['resolution'] if 'resolution' in kwargs else None
        self.is_modified = kwargs['is_modified'] if 'is_modified' in kwargs else None
        self.id = kwargs['id'] if 'id' in kwargs else None
        self.split = kwargs['split'] if 'split' in kwargs else None
        self.add_sos_eos = kwargs['add_sos_eos'] if 'add_sos_eos' in kwargs else False
        if 'openmm_forcefields' in kwargs and kwargs['openmm_forcefields'] is not None:
            self.openmm_forcefields = kwargs['openmm_forcefields']
        else:
            self.openmm_forcefields = OPENMM_FORCEFIELDS

        # Prepare data for model training
        self.int_seq = VOCAB.str2ints(self.seq, add_sos_eos=self.add_sos_eos)
        self.int_mask = [1 if m == "+" else 0 for m in self.mask]
        dssp_vocab = DSSPVocabulary()
        self.int_secondary = dssp_vocab.str2ints(
            self.secondary_structure, add_sos_eos=self.add_sos_eos
        ) if self.secondary_structure is not None else None

        self.sb = None
        self.has_hydrogens = False
        self.openmm_initialized = False
        self.is_numpy = isinstance(self.coords, np.ndarray)
        if self.is_numpy:
            self.hcoords = self.coords.copy() if self.coords is not None else None
        else:
            self.hcoords = torch.clone(self.coords) if self.coords is not None else None
        self.starting_energy = None
        self.positions = None
        self.forces = None
        self._hcoord_mask = None
        self.device = 'cpu'
        self._hcoords_for_openmm = None

    @classmethod
    def from_pkl(cls, pkl_file):
        """Create a SCNProtein from a pickle file (inverse of SCNProtein.to_pkl)."""
        with open(pkl_file, "rb") as f:
            datadict = pickle.load(f)
        return cls(**datadict)

    @classmethod
    def from_pdb(cls, filename, chid=None, pdbid="", include_resolution=False, allow_nan=False):
        """Create a SCNProtein from a PDB file. Warning: does not support gaps.

        Args:
            filename (str): Path to existing PDB file.
            pdbid (str): 4-letter string representing the PDB Identifier.
            include_resolution (bool, default=False): If True, query the PDB for the protein
                structure resolution based off of the given pdb_id.

        Returns:
            A SCNProtein object containing the coorinates, angles, and sequence parsed
            from the PDB file.
        """
        # TODO: Raise an alarm if the user is working with files that have gaps
        # First, use Prody to parse the PDB file
        chain = prody.parsePDB(filename, chain=chid)
        # Next, use SidechainNet to make the relevant measurements given the Prody chain obj
        (dihedrals_np, coords_np, observed_sequence, unmodified_sequence,
         is_nonstd) = scn.utils.measure.get_seq_coords_and_angles(chain,
                                                                  replace_nonstd=True, allow_nan=allow_nan)
        scndata = {
            "coordinates": coords_np.reshape(len(observed_sequence), -1, 3),
            "angles": dihedrals_np,
            "sequence": observed_sequence,
            "unmodified_seq": unmodified_sequence,
            "mask": "+" * len(observed_sequence),
            "is_modified": is_nonstd,
            "id": pdbid,
        }
        # If requested, look up the resolution of the given PDB ID
        if include_resolution:
            assert pdbid, "You must provide a PDB ID to look up the resolution."
            scndata['resolution'] = get_resolution_from_pdbid(pdbid)
        return cls(**scndata)

    @classmethod
    def from_cif(cls, filename, chid=None, pdbid="", include_resolution=False, return_sequences=False):
        """Create a SCNProtein from a mmCIF file. Warning: does not support gaps.

        Args:
            filename (str): Path to existing PDB file.
            pdbid (str): 4-letter string representing the PDB Identifier.
            include_resolution (bool, default=False): If True, query the PDB for the protein
                structure resolution based off of the given pdb_id.

        Returns:
            A SCNProtein object containing the coorinates, angles, and sequence parsed
            from the PDB file.
        """
        # TODO: Raise an alarm if the user is working with files that have gaps
        # First, use Prody to parse the PDB file
        # header = None
        chain, header = prody.parseMMCIF(filename, chain=chid, header=True)
        # chain = prody.parseMMCIF(filename, chain=chid, header=False)
        # Next, use SidechainNet to make the relevant measurements given the Prody chain obj
        (dihedrals_np, coords_np, observed_sequence, unmodified_sequence,
         is_nonstd) = scn.utils.measure.get_seq_coords_and_angles(chain,
                                                                  replace_nonstd=True)
        if not return_sequences and chid is not None and len(observed_sequence) < len(header[chid].sequence):
            raise ValueError("The observed sequence is shorter than the sequence in the "
                             "header. There are likely gaps in the sequence, and this is unsupported.")
        scndata = {
            "coordinates": coords_np.reshape(len(observed_sequence), -1, 3),
            "angles": dihedrals_np,
            "sequence": observed_sequence,
            "unmodified_seq": unmodified_sequence,
            "mask": "+" * len(observed_sequence),
            "is_modified": is_nonstd,
            "id": pdbid,
        }
        # If requested, look up the resolution of the given PDB ID
        if include_resolution:
            assert pdbid, "You must provide a PDB ID to look up the resolution."
            scndata['resolution'] = get_resolution_from_pdbid(pdbid)
        if header is not None and return_sequences:
            return cls(**scndata), observed_sequence, header[chid].sequence
        return cls(**scndata)

    @property
    def sequence(self):
        """Return the protein's sequence in 1-letter amino acid codes."""
        return self.seq

    def __len__(self):
        """Return length of protein sequence."""
        return len(self.seq)

    def to_3Dmol(self, from_angles=False, style=None, other_protein=None):
        """Return an interactive visualization of the protein with py3DMol."""
        if self.sb is None:
            if from_angles:
                self.sb = sidechainnet.StructureBuilder(self.seq,
                                                        self.angles,
                                                        has_hydrogens=self.has_hydrogens)
            elif self.has_hydrogens:
                self.sb = sidechainnet.StructureBuilder(self.seq,
                                                        self.hcoords,
                                                        has_hydrogens=self.has_hydrogens)
            else:
                self.sb = sidechainnet.StructureBuilder(self.seq,
                                                        self.coords,
                                                        has_hydrogens=self.has_hydrogens)
        return self.sb.to_3Dmol(style=style, other_protein=other_protein)

    def to_pdb(self, path, title=None, from_angles=False):
        """Save structure to path as a PDB file."""
        if not title:
            title = self.id
        if self.sb is None:
            if from_angles:
                self.sb = sidechainnet.StructureBuilder(self.seq,
                                                        self.angles,
                                                        has_hydrogens=self.has_hydrogens)
            elif self.has_hydrogens:
                self.sb = sidechainnet.StructureBuilder(self.seq,
                                                        self.hcoords,
                                                        has_hydrogens=self.has_hydrogens)
            else:
                self.sb = sidechainnet.StructureBuilder(self.seq,
                                                        self.coords,
                                                        has_hydrogens=self.has_hydrogens)
        return self.sb.to_pdb(path, title)

    def to_pdbstr(self, title=None, from_angles=False, hcoords=None):
        """Save structure to path as a PDB file."""
        if not title:
            title = self.id
        if hcoords is not None:
            self.sb = sidechainnet.StructureBuilder(self.seq,
                                                    hcoords,
                                                    has_hydrogens=self.has_hydrogens)
        if self.sb is None:
            if from_angles:
                self.sb = sidechainnet.StructureBuilder(self.seq,
                                                        self.angles,
                                                        has_hydrogens=self.has_hydrogens)
            elif self.has_hydrogens:
                self.sb = sidechainnet.StructureBuilder(self.seq,
                                                        self.hcoords,
                                                        has_hydrogens=self.has_hydrogens)
            else:
                self.sb = sidechainnet.StructureBuilder(self.seq,
                                                        self.coords,
                                                        has_hydrogens=self.has_hydrogens)
        return self.sb.to_pdbstr(title)

    def to_gltf(self, path, title="test", from_angles=False):
        """Save structure to path as a gltf (3D-object) file."""
        if self.sb is None:
            if from_angles:
                self.sb = sidechainnet.StructureBuilder(self.seq,
                                                        self.angles,
                                                        has_hydrogens=self.has_hydrogens)
            elif self.has_hydrogens:
                self.sb = sidechainnet.StructureBuilder(self.seq,
                                                        self.hcoords,
                                                        has_hydrogens=self.has_hydrogens)
            else:
                self.sb = sidechainnet.StructureBuilder(self.seq,
                                                        self.coords,
                                                        has_hydrogens=self.has_hydrogens)
        return self.sb.to_gltf(path, title)

    def to_png(self, path, from_angles=False):
        """Save structure to path as a PNG (image) file."""
        if self.sb is None:
            if from_angles:
                self.sb = sidechainnet.StructureBuilder(self.seq,
                                                        self.angles,
                                                        has_hydrogens=self.has_hydrogens)
            elif self.has_hydrogens:
                self.sb = sidechainnet.StructureBuilder(self.seq,
                                                        self.hcoords,
                                                        has_hydrogens=self.has_hydrogens)
            else:
                self.sb = sidechainnet.StructureBuilder(self.seq,
                                                        self.coords,
                                                        has_hydrogens=self.has_hydrogens)
        return self.sb.to_png(path)

    def rmsd(self, other_scnprotein):
        """Compute all-atom RMSD between two sequence-identical SCNProteins."""
        assert self.seq == other_scnprotein.seq, "Proteins must have identical sequences."
        unrolled_coords_a = self.coords.reshape(-1, 3)
        unrolled_coords_b = other_scnprotein.coords.reshape(-1, 3)
        if torch.is_tensor(unrolled_coords_a):
            unrolled_coords_a = unrolled_coords_a.cpu().detach().numpy()
        if torch.is_tensor(unrolled_coords_b):
            unrolled_coords_b = unrolled_coords_b.cpu().detach().numpy()
        real_valued_rows_a = (~np.isnan(unrolled_coords_a)).any(axis=-1)
        real_valued_rows_b = (~np.isnan(unrolled_coords_b)).any(axis=-1)
        a = unrolled_coords_a[real_valued_rows_a & real_valued_rows_b]
        b = unrolled_coords_b[real_valued_rows_a & real_valued_rows_b]
        t = prody.calcTransformation(a, b)
        return prody.calcRMSD(t.apply(a), b)
    
    def get_ca_coords(self):
        """Return a numpy array of the C-alpha coordinates."""
        if torch.is_tensor(self.coords):
            return self.coords[:, 1, :].cpu().detach().numpy()
        else:
            return self.coords[:, 1, :]
    
    def rmsd_ca(self, other_scnprotein):
        """Compute C-alpha RMSD between two sequence-identical SCNProteins."""
        ca1 = self.get_ca_coords()
        ca2 = other_scnprotein.get_ca_coords()
        real_valued_rows_a = (~np.isnan(ca1)).all(axis=-1)
        real_valued_rows_b = (~np.isnan(ca2)).all(axis=-1)
        a = ca1[real_valued_rows_a & real_valued_rows_b]
        b = ca2[real_valued_rows_a & real_valued_rows_b]
        t = prody.calcTransformation(a, b)
        return prody.calcRMSD(t.apply(a), b)

    @property
    def unpadded_coords(self):
        """Return unmasked coordinates."""
        return self.get_unpadded_coords()

    @property
    def num_missing(self):
        """Return number of missing residues."""
        return self.mask.count("-")

    @property
    def seq3(self):
        """Return 3-letter amino acid sequence for the protein."""
        return " ".join([ONE_TO_THREE_LETTER_MAP[c] for c in self.seq])

    def __repr__(self) -> str:
        """Represent an SCNProtein as a string."""
        return (f"SCNProtein({self.id}, len={len(self)}, missing={self.num_missing}, "
                f"split='{self.split}')")

    def fastbuild(self, add_hydrogens=False, build_params=None, inplace=False):
        """Build protein coordinates from angles."""
        if self.is_numpy:
            self.torch()
        coords, params = fastbuild.make_coords(self.seq,
                                               self.angles,
                                               build_params=build_params,
                                               add_hydrogens=add_hydrogens)

        if inplace:
            self.coords = self.hcoords = coords
            self.has_hydrogens = add_hydrogens
            self.sb = None
        else:
            self.sb = sidechainnet.StructureBuilder(self.seq,
                                                    coords,
                                                    has_hydrogens=add_hydrogens)

        return coords

    def build_coords_from_angles(self, angles=None, add_hydrogens=False):
        """Build protein coordinates iff no StructureBuilder already exists."""
        raise ValueError("Use fastbuild instead.")

    def add_hydrogens(self, add_to_heavy_atoms=False):
        """Add hydrogens to the internal protein structure representation."""
        if not add_to_heavy_atoms:
            raise ValueError(
                "Adding hydrogens without add_to_heavy_atoms==True is not "
                "supported.\nDo you want to use fastbuild to build all atoms from angles "
                "instead?")
        starting_is_numpy = self.is_numpy
        self.torch()
        self.cuda()
        self.hb = sidechainnet.structure.HydrogenBuilder.HydrogenBuilder(
            self.seq, self.coords.double(), device=self.device)
        self.hcoords = self.hb.build_hydrogens()
        self.has_hydrogens = True
        if starting_is_numpy:
            self.numpy()

    def get_unpadded_coords(self):
        """Get coordinates without padding and without ATOMS_PER_RES dimension."""
        unrolled_coords = self.coords.reshape(-1, 3)
        real_valued_rows = (~torch.isnan(unrolled_coords)).any(dim=-1)
        return unrolled_coords[real_valued_rows]

    ##########################################
    #        OPENMM PRIMARY FUNCTIONS        #
    ##########################################

    def _convert_pdbfixer_positions_to_quantities(self):
        """Convert positions to quantities."""
        self.pdbfixer.positions *= unit.nanometers

    def _undo_pdbfixer_positions_to_quantities(self):
        """Unconvert positions to quantities."""
        self.pdbfixer.positions /= unit.nanometers

    def _add_missing_via_pdbfixer_and_init_openmm(self,
                                                  seed=SEED,
                                                  add_hydrogens_via_openmm=False):
        self.get_openmm_repr()
        self.make_pdbfixer()
        self._convert_pdbfixer_positions_to_quantities()
        self.pdbfixer.findMissingResidues()
        self.pdbfixer.findMissingAtoms()
        self.pdbfixer.addMissingAtoms()
        self.pdbfixer.addMissingHydrogens(pH=7.0,
                                          forcefield=ForceField(*self.openmm_forcefields))
        self._undo_pdbfixer_positions_to_quantities()
        self.positions = self.pdbfixer.positions
        self.topology = self.pdbfixer.topology
        self.initialize_openmm(skip_get_openmm_repr=True,
                               add_hydrogens_via_openmm=add_hydrogens_via_openmm)

    def get_energy(self,
                   add_missing=False,
                   add_hydrogens_via_openmm=False,
                   add_hydrogens_via_scnprotein=False,
                   return_unitless_kjmol=False):
        """Return potential energy of the system given current atom positions."""
        if not self.has_hydrogens and not (add_hydrogens_via_openmm or add_missing or add_hydrogens_via_scnprotein):
            raise ValueError("Cannot compute energy without hydrogens.")
        if add_hydrogens_via_openmm and add_hydrogens_via_scnprotein:
            raise ValueError("Cannot add hydrogens via both OpenMM and SCNProtein.")
        if add_missing:
            try:
                self._add_missing_via_pdbfixer_and_init_openmm(
                    seed=SEED, add_hydrogens_via_openmm=add_hydrogens_via_openmm)
            except openmm.OpenMMException as e:
                if "Particle coordinate is nan" in str(e):
                    # The system exploded when adding atoms. Try a different seed.
                    self._add_missing_via_pdbfixer_and_init_openmm(
                        seed=SEED + 1, add_hydrogens_via_openmm=add_hydrogens_via_openmm)
        if add_hydrogens_via_scnprotein:
            self.add_hydrogens(add_to_heavy_atoms=True)

        if not self.openmm_initialized:
            self.initialize_openmm(add_hydrogens_via_openmm=add_hydrogens_via_openmm)

        self.simulation.context.setPositions(self.positions)
        self.starting_state = self.simulation.context.getState(getEnergy=True,
                                                               getForces=True)
        self.starting_energy = self.starting_state.getPotentialEnergy()
        if return_unitless_kjmol:
            return self.starting_energy.value_in_unit(unit.kilojoule_per_mole)
        else:
            return self.starting_energy

    def get_energy_loss(self, nonbonded_interactions=True):
        """Return potential energy loss of the system given current atom positions."""
        if not self.openmm_initialized or not nonbonded_interactions:
            self.initialize_openmm(nonbonded_interactions=nonbonded_interactions)
        eloss_fn = OpenMMEnergyH()
        eloss = eloss_fn.apply(self, self.hcoords)
        return eloss

    def get_forces(self, pprint=False):
        """Return tensor of forces as requested."""
        # Initialize tensor and energy
        if self.forces is None:
            self.forces = np.zeros((len(self.seq) * NUM_COORDS_PER_RES_W_HYDROGENS, 3))
        if not self.starting_energy:
            self.get_energy()

        # Compute forces with OpenMM, but convert to a value with angstroms and kJ/mol
        self._forces_raw = self.starting_state.getForces(asNumpy=True).value_in_unit(
            kilojoule / (angstrom * mole))

        # Assign forces from OpenMM to their places in the hydrogen coord representation
        self.forces[self.hcoord_to_pos_map_keys] = self._forces_raw[
            self.hcoord_to_pos_map_values]

        if pprint:
            atom_name_pprint(self.get_atom_names(heavy_only=False), self.forces)
            # return

        return self.forces

    def update_hydrogens(self, hcoords):
        """Take a set of hydrogen coordinates and use it to update this protein."""
        mask = self.get_hydrogen_coord_mask()
        self.hcoords = hcoords * mask
        self.has_hydrogens = True
        self.update_positions()

    def update_hydrogens_for_openmm(self, hcoords):
        """Use a set of hydrogen coords to update OpenMM data for this protein."""
        if not self.openmm_initialized:
            self.initialize_openmm()
        #     self.update_positions()

        mask = self.get_hydrogen_coord_mask()
        self._hcoords_for_openmm = torch.nan_to_num(hcoords * mask, nan=0.0)
        self.update_positions(self._hcoords_for_openmm)  # Use our openmm only hcoords

        # Below is experimental code that only passes hcoords from GPU to CPU once
        # If this code is in use, then the coordinates do not update. If the above line
        # of code is in use, the the coordinates will update, but must be passed from GPU
        # to CPU each time.

        # if not hasattr(self, "_hcoords_for_openmm_cpu"):
        #     self._hcoords_for_openmm_cpu = self._hcoords_for_openmm.cpu().detach().numpy()
        # self.update_positions(self._hcoords_for_openmm_cpu)

    def update_positions(self, hcoords=None):
        """Update the positions variable with hydrogen coordinate values."""
        if hcoords is None:
            hcoords = self.hcoords
        # The below step takes a PyTorch CUDA representation of all-atom coordinates
        # and passes it to the CPU as a numpy array so that OpenMM can read it
        hcoords = hcoords.detach().cpu().numpy() if not isinstance(
            hcoords, np.ndarray) else hcoords
        # We must also convert hcoords from Angstroms to nanometers for Openmm
        hcoords = hcoords / 10.0
        # A mapping is used to define the relationship between hcoords and positions
        self.positions[self.hcoord_to_pos_map_values] = hcoords.reshape(
            -1, 3)[self.hcoord_to_pos_map_keys]
        return self.positions  # TODO numba JIT compile

    ##########################################
    #         OPENMM SETUP FUNCTIONS         #
    ##########################################

    def get_openmm_repr(self, skip_missing_residues=True):
        """Return tuple of OpenMM topology and positions for analysis with OpenMM."""
        self.hcoord_to_pos_map_keys = []
        self.hcoord_to_pos_map_values = []
        pos_idx = hcoord_idx = 0
        self.positions = []
        self.topology = Topology()
        self.openmm_seq = ""
        chain = self.topology.addChain()
        hcoords = self.hcoords.cpu().detach().numpy(
        ) if not self.is_numpy else self.hcoords
        # hcoords are always stored in Angstroms, but OpenMM wants nanometers
        hcoords = hcoords / 10.0
        coord_gen = coord_generator(hcoords)
        self.has_missing_atoms = False
        for i, (residue_code, coords, mask_char, atom_names) in enumerate(
                zip(self.seq, coord_gen, self.mask, self.get_atom_names())):
            residue_name = ONE_TO_THREE_LETTER_MAP[residue_code]
            if mask_char == "-" and skip_missing_residues:
                hcoord_idx += NUM_COORDS_PER_RES_W_HYDROGENS
                continue
            residue = self.topology.addResidue(name=residue_name, chain=chain)

            for j, (an, c) in enumerate(zip(atom_names, coords)):
                # If this atom is a PAD character or non-existent terminal atom, skip
                # TODO more graciously handle pads for terminal residues
                if an == "PAD" or (an in ["OXT", "H2", "H3"]
                                   and np.isnan(c).any()) or (residue_code == 'P'
                                                              and an == 'H'):
                    hcoord_idx += 1
                    continue
                # Handle missing atoms
                if np.isnan(c).any():
                    # print(c)
                    raise ValueError("Cannot construct an OpenMM Representation with "
                                     f"missing atoms ({i} {residue_name} {self}).")
                    self.has_missing_atoms = True
                    continue
                self.topology.addAtom(name=an,
                                      element=get_element_from_atomname(an),
                                      residue=residue)
                self.positions.append(c)
                self.hcoord_to_pos_map_keys.append(hcoord_idx)
                self.hcoord_to_pos_map_values.append(pos_idx)
                hcoord_idx += 1
                pos_idx += 1

            self.openmm_seq += residue_code

        self.topology.createStandardBonds()
        # TODO think about disulfide bonds at a later point, see CYS/CYX (bridge, no H)
        # self.topology.createDisulfideBonds(self.positions)
        self.positions = np.array(self.positions)
        return self.topology, self.positions

    def initialize_openmm(self,
                          nonbonded_interactions=True,
                          skip_get_openmm_repr=False,
                          add_hydrogens_via_openmm=False):
        """Create top., pos., modeller, forcefield, system, integrator, & simulation."""
        if not skip_get_openmm_repr:
            self.get_openmm_repr()
        self.modeller = Modeller(self.topology, self.positions)
        self.forcefield = ForceField(*self.openmm_forcefields)
        if add_hydrogens_via_openmm:
            random.seed(SEED)
            self.modeller.addHydrogens(forcefield=self.forcefield)
            self.positions = self.modeller.positions
        elif not self.has_hydrogens:
            self.add_hydrogens(add_to_heavy_atoms=True)
        self.system = self.forcefield.createSystem(self.modeller.topology,
                                                   nonbondedMethod=openmm.app.NoCutoff,
                                                   nonbondedCutoff=1 * nanometer,
                                                   constraints=HBonds)
        self.integrator = LangevinMiddleIntegrator(300 * kelvin, 1 / picosecond,
                                                   0.004 * picoseconds)
        if not nonbonded_interactions:
            print(f"Removing {self.system.getForce(3)}.")
            self.system.removeForce(3)
        if OPENMM_PLATFORM == "CUDA":
            self.platform = Platform.getPlatformByName('CUDA')
            properties = {'DeviceIndex': '0', 'Precision': 'double'}
        else:
            self.platform = Platform.getPlatformByName('CPU')
            properties = {}
        self.simulation = openmm.app.Simulation(self.modeller.topology,
                                                self.system,
                                                self.integrator,
                                                platform=self.platform,
                                                platformProperties=properties)
        self.openmm_initialized = True
        self.simulation.context.setPositions(self.modeller.positions)

    def get_hydrogen_coord_mask(self, zeros_instead_of_nans=False):
        """Return a torch tensor with nans representing pad characters in self.hcoords."""
        if self._hcoord_mask is not None:
            return self._hcoord_mask
        else:
            self._hcoord_mask = torch.full_like(self.hcoords, 1)
            for i, (res3, atom_names) in enumerate(
                    zip(self.seq3.split(" "), self.get_atom_names())):
                pad_chars = [an == "PAD" for an in atom_names]
                if i == 0:
                    # Terminal does not have OXT
                    self._hcoord_mask[i, 4, :] = torch.nan
                if i == self.hcoords.shape[0] - 1:
                    # Terminal does not have H2/H3
                    # TODO improve atom naming convention
                    self._hcoord_mask[i, [6, 7], :] = torch.nan
                if res3 == "PRO":
                    # Proline does not have H
                    self._hcoord_mask[i, 5, :] = torch.nan
                self._hcoord_mask[i, pad_chars, :] = torch.nan
            if zeros_instead_of_nans:
                self._hcoord_mask[torch.isnan(self._hcoord_mask)] = 0
            self._hcoord_mask = self._hcoord_mask.type(torch.bool)
            return self._hcoord_mask

    ##########################################
    #         OTHER OPENMM FUNCTIONS         #
    ##########################################

    def make_pdbfixer(self):
        """Construct and return an OpenMM PDBFixer object for this protein."""
        from pdbfixer import PDBFixer
        self.pdbfixer = PDBFixer(topology=self.topology,
                                 positions=self.positions,
                                 sequence=self.openmm_seq)
        return self.pdbfixer

    def run_pdbfixer(self):
        """Add missing atoms to protein's PDBFixer representation."""
        self.pdbfixer.findMissingResidues()
        self.pdbfixer.findMissingAtoms()
        # print("Missing atoms", self.pdbfixer.missingAtoms)
        self.pdbfixer.findNonstandardResidues()
        self.pdbfixer.addMissingAtoms()
        self.pdbfixer.addMissingHydrogens(7.0)

    def minimize(self, nonbonded_interactions=True, add_hydrogens_via_openmm=False):
        """Perform an energy minimization using the PDBFixer representation. Return ∆E."""
        assert self.has_hydrogens, "You must add hydrogens before minimizing. (self.fastbuild, self.add_hydrogens)"
        cutoff_method = openmm.app.NoCutoff if nonbonded_interactions else openmm.app.CutoffNonPeriodic
        cutoff_dist = 1 * nanometer if nonbonded_interactions else 1e-5 * nanometer
        constraints = HBonds if nonbonded_interactions else None
        if not hasattr(self, "topology") or self.topology is None:
            self.initialize_openmm(add_hydrogens_via_openmm=add_hydrogens_via_openmm)
        if not self.has_hydrogens:
            self.add_hydrogens(add_to_heavy_atoms=True)
        self.modeller = Modeller(self.topology, self.positions)
        self.forcefield = ForceField(*self.openmm_forcefields)
        self.system = self.forcefield.createSystem(self.modeller.topology,
                                                   nonbondedMethod=cutoff_method,
                                                   nonbondedCutoff=cutoff_dist,
                                                   constraints=constraints)
        if not nonbonded_interactions:
            print(f"Removing {self.system.getForce(3)}.")
            self.system.removeForce(3)
        self.integrator = LangevinMiddleIntegrator(300 * kelvin, 1 / picosecond,
                                                   0.004 * picoseconds)
        self.simulation = openmm.app.Simulation(self.modeller.topology, self.system,
                                                self.integrator)
        self.simulation.context.setPositions(self.modeller.positions)
        self.starting_energy = self.simulation.context.getState(
            getEnergy=True).getPotentialEnergy()
        self.starting_positions = np.asarray(
            self.simulation.context.getState(getPositions=True).getPositions(
                asNumpy=True))
        self.simulation.minimizeEnergy()
        self.state = self.simulation.context.getState(getVelocities=True,
                                                      getPositions=True,
                                                      getParameters=True,
                                                      getEnergy=True,
                                                      getForces=True)
        self.ending_energy = self.state.getPotentialEnergy()
        self.ending_positions = np.asarray(self.state.getPositions(asNumpy=True))
        self.update_hcoords_from_openmm_positions(self.ending_positions)
        return self.ending_energy - self.starting_energy
    
    def update_hcoords_from_openmm_positions(self, openmm_positions):
        """Update self.hcoords from OpenMM positions."""
        self.has_hydrogens = True
        # create a tensor filled with nans of the correct shape
        self.hcoords = torch.full_like(self.hcoords, float("nan"))
        # fill in the positions
        self.hcoords[self.get_hydrogen_coord_mask(zeros_instead_of_nans=True)] = torch.from_numpy(openmm_positions)
        self.positions[self.hcoord_to_pos_map_values] = self.hcoords.reshape(
            -1, 3)[self.hcoord_to_pos_map_keys]
        self.coords = self.hcoords


    def write_ending_positions_to_pdbfile(self, filename):
        with open(filename, 'w') as f:
            PDBFile.writeFile(self.simulation.topology, self.ending_positions * 10, f)

    def get_energy_difference(self):
        """Create PDBFixer object, minimize, and report ∆E."""
        self.add_hydrogens()
        self.topology, self.positions = self.get_openmm_repr()
        return self.minimize()

    def get_rmsd_difference(self):
        """Report difference in start/end coordinates after energy minimization."""
        aligned_minimized = prody.calcTransformation(
            self.ending_positions, self.starting_positions).apply(self.ending_positions)
        rmsd = prody.calcRMSD(self.starting_positions, aligned_minimized)
        return rmsd

    ###################################
    #         OTHER FUNCTIONS         #
    ###################################

    def get_atom_names(self, zip_coords=False, pprint=False, heavy_only=False):
        """Return or print atom name list for each residue including terminal atoms."""
        all_atom_name_list = []
        for i in range(len(self.seq)):
            if not self.has_hydrogens or heavy_only:
                atom_names = ATOM_MAP_HEAVY[self.seq[i]]
            else:
                atom_names = ATOM_MAP_H[self.seq[i]]
            all_atom_name_list.append(atom_names)

        if pprint:
            _size = self.coords.shape[1]
            nums = list(range(_size))
            print(" ".join([f"{n: <4}" for n in nums]))
            for ans in all_atom_name_list:
                ans = [f"{a: <4}" for a in ans]
                print(" ".join(ans))
            return None

        # Prints a representation of coordinates and their atom names
        if zip_coords:
            flat_list = [
                f"{item: <4}" for sublist in all_atom_name_list for item in sublist
            ]
            if heavy_only:
                items = list(zip(flat_list, self.coords.reshape(-1, 3)))
            elif self.has_hydrogens:
                items = list(zip(flat_list, self.hcoords.reshape(-1, 3)))
            else:
                items = list(zip(flat_list, self.coords.reshape(-1, 3)))
            for i, xy in enumerate(items):
                xy = list(xy)
                xy[1] = xy[1].detach().numpy()
                print(f"{i: <2}", xy[0], xy[1])
            return None
        return all_atom_name_list

    def insert_terminal_atoms_into_name_list(self, atom_names, terminal_atom_names):
        """Insert a list of atoms into an existing list, keeping lengths the same."""
        pad_idx = atom_names.index("PAD")
        new_list = list(atom_names)
        new_list[pad_idx:pad_idx + len(terminal_atom_names)] = terminal_atom_names
        return new_list

    def hydrogenrep_to_heavyatomrep(self, hcoords=None, inplace=False):
        """Remove hydrogens from a tensor of coordinates in heavy atom representation."""
        if hcoords is None:
            hcoords = self.hcoords

        if not self.is_numpy:
            new_coords = torch.zeros(len(hcoords), NUM_COORDS_PER_RES, 3) * GLOBAL_PAD_CHAR
        else:
            new_coords = np.zeros((len(hcoords), NUM_COORDS_PER_RES, 3)) * GLOBAL_PAD_CHAR

        mask = HEAVY_ATOM_MASK_TENSOR[self.int_seq]
        for i, (resmask, res) in enumerate(zip(mask, hcoords)):
            newres = res[resmask.bool()]
            new_coords[i, :len(newres)] = newres

        if inplace:
            self.hcoords = self.coords = new_coords
            self.sb = None
            self.has_hydrogens = False

        return new_coords

    def cuda(self):
        """Move coords, hcoords, and angles to the default CUDA torch device."""
        if isinstance(self.coords, torch.Tensor):
            self.coords = self.coords.cuda()
        else:
            self.coords = torch.tensor(self.coords, device='cuda')
        if isinstance(self.hcoords, torch.Tensor):
            self.hcoords = self.hcoords.cuda()
        else:
            self.hcoords = torch.tensor(self.hcoords, device='cuda')
        if self.angles is not None and isinstance(self.angles, torch.Tensor):
            self.angles = self.angles.cuda()
        elif self.angles is not None:
            self.angles = torch.tensor(self.angles, device='cuda')
        self.device = 'cuda'
        self.is_numpy = False

    def cpu(self):
        """Move coords, hcoords, and angles to the default CUDA torch device."""
        if isinstance(self.coords, torch.Tensor):
            self.coords = self.coords.cpu()
        else:
            self.coords = torch.tensor(self.coords, device='cpu')
        if isinstance(self.hcoords, torch.Tensor):
            self.hcoords = self.hcoords.cpu()
        else:
            self.hcoords = torch.tensor(self.hcoords, device='cpu')
        if isinstance(self.angles, torch.Tensor):
            self.angles = self.angles.cpu()
        else:
            self.angles = torch.tensor(self.angles, device='cpu')
        self.device = 'cpu'
        self.is_numpy = False

    def numpy(self):
        """Change coords, hcoords, and angles to numpy.ndarray objects."""
        if not isinstance(self.coords, np.ndarray):
            self.coords = self.coords.cpu().detach().numpy()
        if not isinstance(self.hcoords, np.ndarray):
            self.hcoords = self.hcoords.cpu().detach().numpy()
        if not isinstance(self.angles, np.ndarray):
            self.angles = self.angles.cpu().detach().numpy()
        self.is_numpy = True

    def torch(self):
        """Change coords, hcoords, and angles to torch.tensor objects."""
        if not torch.is_tensor(self.coords) and self.coords is not None:
            self.coords = torch.tensor(self.coords)
        if not torch.is_tensor(self.hcoords) and self.hcoords is not None:
            self.hcoords = torch.tensor(self.hcoords)
        if not torch.is_tensor(self.angles) and self.angles is not None:
            self.angles = torch.tensor(self.angles)
        self.is_numpy = False

    def fillna(self, value=0.0, warn=True):
        """Replace nans in coordinate and angle matrices with the specified value.

        Args:
            value (float, optional): Replace nans with this value. Defaults to 0.0.
        """
        if warn:
            warnings.warn(
                "Doing this will remove all nans from the object and may cause missing"
                " residue information to be lost. To proceed, call with warn=False.")
            return
        self.coords[np.isnan(self.coords)] = value
        self.angles[np.isnan(self.angles)] = value

    def pickle(self, path):
        """Write a pickled version of the protein.

        Args:
            path (str): Path to pickle file.
        """
        if self.has_hydrogens:
            self.hcoords = torch.tensor(self.hcoords)
            self.coords = self.hydrogenrep_to_heavyatomrep().detach().numpy()
        d = {
            "angles": self.angles,
            "sequence": self.seq,
            "id": self.id,
            "evolutionary": self.evolutionary,
            "mask": self.mask,
            "coordinates": self.coords,
            "secondary_structure": self.secondary_structure,
            "resolution": self.resolution,
            "unmodified_seq": self.unmodified_seq,
            "is_modified": self.is_modified,
            "split": self.split,
            "add_sos_eos": self.add_sos_eos
        }
        with open(path, "wb") as f:
            pickle.dump(d, f)

    def copy(self):
        """Duplicates the protein. Does not support OpenMM data."""
        newp = SCNProtein(
            coordinates=self.coords.copy() if self.is_numpy else self.coords.clone(),
            angles=self.angles.copy() if self.is_numpy else self.coords.clone(),
            sequence=self.seq,
            unmodified_seq=self.unmodified_seq,
            mask=self.mask,
            evolutionary=self.evolutionary,
            secondary_structure=self.secondary_structure,
            resolution=self.resolution,
            is_modified=self.is_modified,
            id=self.id,
            split=self.split,
            add_sos_eos=self.add_sos_eos)
        newp.hcoords = self.hcoords.copy() if self.is_numpy else self.coords.clone()
        newp.has_hydrogens = self.has_hydrogens
        return newp

    def trim_edges(self):
        """Trim edges of seq/ums, 2ndary, evo, mask, angles, and coords based on mask."""
        assert isinstance(self.mask, str)

        mask_seq_no_left = self.mask.lstrip('-')
        mask_seq_no_right = self.mask.rstrip('-')
        n_removed_left = len(self.mask) - len(mask_seq_no_left)
        n_removed_right = len(self.mask) - len(mask_seq_no_right)
        n_removed_right = None if n_removed_right == 0 else -n_removed_right
        # Trim simple attributes
        for at in [
                "seq", "int_seq", "angles", "secondary_structure", "int_secondary",
                "is_modified", "evolutionary", "int_mask", "mask"
        ]:
            data = getattr(self, at)
            if data is not None:
                setattr(self, at, data[n_removed_left:n_removed_right])
        # Trim coordinate data
        assert len(self.coords.shape) == 3, 'Must have coords shape like (LxNx3).'
        if self.coords is not None:
            self.coords = self.coords[n_removed_left:n_removed_right]
        if self.hcoords is not None:
            self.hcoords = self.hcoords[n_removed_left:n_removed_right]
        # Trim unmodified seq
        if self.unmodified_seq is not None:
            assert isinstance(self.unmodified_seq, str)
            ums = self.unmodified_seq.split()
            self.unmodified_seq = " ".join(ums[n_removed_left:n_removed_right])
        # Reset structure builders
        self.sb = None

    def trim_to_max_seq_len(self):
        """Trim edges of seq/ums, 2ndary, evo, mask, angles, and coords to MAX_SEQ_LEN."""
        if len(self) <= MAX_SEQ_LEN:
            return
        n_removed_right = MAX_SEQ_LEN - len(self.seq)
        # Trim simple attributes
        for at in [
                "seq", "int_seq", "angles", "secondary_structure", "int_secondary",
                "is_modified", "evolutionary", "int_mask", "mask"
        ]:
            data = getattr(self, at)
            if data is not None:
                setattr(self, at, data[:n_removed_right])
        # Trim coordinate data
        end_point = n_removed_right
        if self.coords is not None:
            self.coords = self.coords[:end_point]
        if self.hcoords is not None:
            self.hcoords = self.hcoords[:end_point]
        # Trim unmodified seq
        if self.unmodified_seq is not None:
            assert isinstance(self.unmodified_seq, str)
            ums = self.unmodified_seq.split()
            self.unmodified_seq = " ".join(ums[:n_removed_right])
        # Reset structure builders
        self.sb = None
    
    def reset_openmm_data(self):
        """Reset all OpenMM data."""
        self.sb = None
        self.openmm_initialized = False
    
    def reset_hydrogens_and_openmm(self):
        """Reset all hydrogen data."""
        if self.has_hydrogens:
            self.hydrogenrep_to_heavyatomrep(inplace=True)
        else:
            assert self.coords.shape[1] == NUM_COORDS_PER_RES, 'coords must be heavy atom rep.'
            self.hcoords = self.coords
        self.reset_openmm_data()


def atom_name_pprint(atom_names, values):
    """Nicely print atom names and values."""
    flat_list = [item for sublist in atom_names for item in sublist]
    for i, (an, vals) in enumerate(zip(flat_list, values)):
        print(f"{i: <2}", f"{an: <4}", vals)


def get_element_from_atomname(atom_name):
    """Return openmm.app.element object matching a given atom name.

    Args:
        atom_name (str): Atom name (PDB format).

    Raises:
        ValueError: If atom_name is not N, C, O, S, or PAD.

    Returns:
        openmm.app.element: For example nitrogen, carbon, oxygen, sulfur. Returns None
        if atom_name is PAD.
    """
    if atom_name[0] == "N":
        return elem.nitrogen
    elif atom_name[0] == "C":
        return elem.carbon
    elif atom_name[0] == "O":
        return elem.oxygen
    elif atom_name[0] == "S":
        return elem.sulfur
    elif atom_name == "PAD":
        return None
    elif atom_name[0] == "H":
        return elem.hydrogen
    else:
        raise ValueError(f"Unknown element for atom name {atom_name}.")


if __name__ == "__main__":
    import sidechainnet as scn
    d = scn.load("debug",
                 scn_dataset=True,
                 complete_structures_only=True,
                 trim_edges=True,
                 scn_dir="/home/jok120/sidechainnet_data",
                 filter_by_resolution=True)
    d.filter(lambda x: len(x) < 15)
    d[2].add_hydrogens(from_angles=True)
    d[2].get_energy()
