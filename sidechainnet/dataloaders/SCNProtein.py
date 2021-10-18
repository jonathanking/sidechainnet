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

import numpy as np
import openmm
import prody
import torch
from openmm import Platform
from openmm.app import Topology
from openmm.app import element as elem
from openmm.app.forcefield import ForceField, HBonds
from openmm.app.modeller import Modeller
from openmm.openmm import LangevinMiddleIntegrator
from openmm.unit import kelvin, nanometer, picosecond, picoseconds

import sidechainnet
import sidechainnet.structure.HydrogenBuilder as hy
from sidechainnet import structure
from sidechainnet.structure.build_info import NUM_COORDS_PER_RES, SC_BUILD_INFO
from sidechainnet.structure.PdbBuilder import ATOM_MAP_14
from sidechainnet.structure.structure import coord_generator
from sidechainnet.utils.sequence import ONE_TO_THREE_LETTER_MAP

OPENMM_FORCEFIELDS = ['amber14/protein.ff15ipq.xml', 'amber14/spce.xml']


class SCNProtein(object):
    """Represent one protein in SidechainNet. Created programmatically by SCNDataset."""

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.coords = kwargs['coordinates']
        self.angles = kwargs['angles']
        self.seq = kwargs['sequence']
        self.unmodified_seq = kwargs['unmodified_seq']
        self.mask = kwargs['mask']
        self.evolutionary = kwargs['evolutionary']
        self.secondary_structure = kwargs['secondary_structure']
        self.resolution = kwargs['resolution']
        self.is_modified = kwargs['is_modified']
        self.id = kwargs['id']
        self.split = kwargs['split']
        self.sb = None
        self.atoms_per_res = NUM_COORDS_PER_RES
        self.has_hydrogens = False
        self.openmm_initialized = False
        self.hcoords = self.coords.copy()
        self.starting_energy = None
        self.positions = None
        self.forces = None
        self._hcoord_mask = None
        self.device = 'cpu'
        self.is_numpy = False
        self._hcoords_for_openmm = None

    def __len__(self):
        """Return length of protein sequence."""
        return len(self.seq)

    def to_3Dmol(self):
        """Return an interactive visualization of the protein with py3DMol."""
        if self.sb is None:
            if self.has_hydrogens:
                self.sb = sidechainnet.StructureBuilder(self.seq, self.hcoords)
            else:
                self.sb = sidechainnet.StructureBuilder(self.seq, self.coords)
        return self.sb.to_3Dmol()

    def to_pdb(self, path, title=None):
        """Save structure to path as a PDB file."""
        if not title:
            title = self.id
        if self.sb is None:
            if self.has_hydrogens:
                self.sb = sidechainnet.StructureBuilder(self.seq, self.hcoords)
            else:
                self.sb = sidechainnet.StructureBuilder(self.seq, self.coords)
        return self.sb.to_pdb(path, title)

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

    def add_hydrogens(self, from_angles=False, coords=None):
        """Add hydrogens to the internal protein structure representation."""
        if from_angles:
            self.sb = structure.StructureBuilder(self.seq,
                                                 ang=self.angles,
                                                 device=self.device)
            self.sb.build()
        elif coords is not None:
            self.sb = structure.StructureBuilder(self.seq, crd=coords, device=self.device)
        else:
            self.sb = structure.StructureBuilder(self.seq,
                                                 crd=self.coords,
                                                 device=self.device)
        self.sb.add_hydrogens()
        self.hcoords = self.sb.coords
        self.has_hydrogens = True
        self.atoms_per_res = hy.NUM_COORDS_PER_RES_W_HYDROGENS
        if self.positions is None:
            self.initialize_openmm()
        self.update_positions()

    ##########################################
    #        OPENMM PRIMARY FUNCTIONS        #
    ##########################################

    def get_energy(self):
        """Return potential energy of the system given current atom positions."""
        if not self.openmm_initialized:
            self.initialize_openmm()
        self.simulation.context.setPositions(self.positions)
        self.starting_state = self.simulation.context.getState(getEnergy=True,
                                                               getForces=True)
        self.starting_energy = self.starting_state.getPotentialEnergy()
        return self.starting_energy

    def get_forces(self, pprint=False):
        """Return tensor of forces as requested."""
        # Initialize tensor and energy
        if self.forces is None:
            self.forces = np.zeros((len(self.seq) * hy.NUM_COORDS_PER_RES_W_HYDROGENS, 3))
        if not self.starting_energy:
            self.get_energy()

        # Compute forces with OpenMM
        self._forces_raw = self.starting_state.getForces(asNumpy=True)

        # Assign forces from OpenMM to their places in the hydrogen coord representation
        self.forces[self.hcoord_to_pos_map_keys] = self._forces_raw[self.hcoord_to_pos_map_values]

        if pprint:
            atom_name_pprint(self.get_atom_names(heavy_only=False), self.forces)
            return

        return self.forces

    def update_hydrogens(self, hcoords):
        """Take a set of hydrogen coordinates and use it to update this protein."""
        mask = self.get_hydrogen_coord_mask()
        self.hcoords = hcoords * mask
        self.has_hydrogens = True
        self.atoms_per_res = hy.NUM_COORDS_PER_RES_W_HYDROGENS
        self.update_positions()

    def update_hydrogens_for_openmm(self, hcoords):
        """Use a set of hydrogen coords to update OpenMM data for this protein."""
        mask = self.get_hydrogen_coord_mask()
        self._hcoords_for_openmm = hcoords * mask
        self.update_positions(self._hcoords_for_openmm)

    def update_positions(self, hcoords=None):
        """Update the positions variable with hydrogen coordinate values."""
        if hcoords is None:
            hcoords = self.hcoords
        hcoords = hcoords.cpu().detach().numpy() if not self.is_numpy else hcoords
        self.positions[self.hcoord_to_pos_map_values] = hcoords[self.hcoord_to_pos_map_keys]
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
        hcoords = self.hcoords.cpu().detach().numpy() if not self.is_numpy else self.hcoords
        coord_gen = coord_generator(hcoords, self.atoms_per_res)
        for i, (residue_code, coords, mask_char, atom_names) in enumerate(
                zip(self.seq, coord_gen, self.mask, self.get_atom_names())):
            residue_name = ONE_TO_THREE_LETTER_MAP[residue_code]
            if mask_char == "-" and skip_missing_residues:
                hcoord_idx += hy.NUM_COORDS_PER_RES_W_HYDROGENS
                continue
            residue = self.topology.addResidue(name=residue_name, chain=chain)

            for j, (an, c) in enumerate(zip(atom_names, coords)):
                if an == "PAD":
                    hcoord_idx += 1
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

    def initialize_openmm(self):
        """Create top., pos., modeller, forcefield, system, integrator, & simulation."""
        self.get_openmm_repr()
        self.modeller = Modeller(self.topology, self.positions)
        self.forcefield = ForceField(*OPENMM_FORCEFIELDS)
        self.system = self.forcefield.createSystem(self.modeller.topology,
                                                   nonbondedMethod=openmm.app.NoCutoff,
                                                   nonbondedCutoff=1 * nanometer,
                                                   constraints=HBonds)
        self.integrator = LangevinMiddleIntegrator(300 * kelvin, 1 / picosecond,
                                                   0.004 * picoseconds)
        self.platform = Platform.getPlatformByName('CUDA')
        properties = {'DeviceIndex': '0', 'Precision': 'double'}
        self.simulation = openmm.app.Simulation(self.modeller.topology,
                                                self.system,
                                                self.integrator,
                                                platform=self.platform,
                                                platformProperties=properties)
        self.openmm_initialized = True

    def get_hydrogen_coord_mask(self):
        """Return a torch tensor with 0s representing pad characters in self.hcoords."""
        if self._hcoord_mask is not None:
            return self._hcoord_mask
        else:
            self._hcoord_mask = torch.zeros_like(self.hcoords)
            for i, (res3, atom_names) in enumerate(
                    zip(self.seq3.split(" "), self.get_atom_names())):
                n_heavy_atoms = sum([True if an != "PAD" else False for an in atom_names])
                n_hydrogens = len(hy.HYDROGEN_NAMES[res3])
                self._hcoord_mask[i * hy.NUM_COORDS_PER_RES_W_HYDROGENS:i *
                                  hy.NUM_COORDS_PER_RES_W_HYDROGENS + n_heavy_atoms +
                                  n_hydrogens, :] = 1.0
            return self._hcoord_mask

    ##########################################
    #         OTHER OPENMM FUNCTIONS         #
    ##########################################

    def make_pdbfixer(self):
        """Construct and return an OpenMM PDBFixer object for this protein."""
        from pdbfixer import PDBFixer
        self.pdbfixer = PDBFixer(topology=self.topology,
                                 positions=self.positions,
                                 sequence=self.openmm_seq,
                                 use_topology=True)
        return self.pdbfixer

    def run_pdbfixer(self):
        """Add missing atoms to protein's PDBFixer representation."""
        self.pdbfixer.findMissingResidues()
        self.pdbfixer.findMissingAtoms()
        print("Missing atoms", self.pdbfixer.missingAtoms)
        # self.pdbfixer.findNonstandardResidues()
        # self.pdbfixer.addMissingAtoms()
        # self.pdbfixer.addMissingHydrogens(7.0)

    def minimize(self):
        """Perform an energy minimization using the PDBFixer representation. Return ∆E."""
        self.modeller = Modeller(self.topology, self.positions)
        self.forcefield = ForceField(*OPENMM_FORCEFIELDS)
        self.system = self.forcefield.createSystem(self.modeller.topology,
                                                   nonbondedMethod=openmm.app.NoCutoff,
                                                   nonbondedCutoff=1 * nanometer,
                                                   constraints=HBonds)
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
        return self.ending_energy - self.starting_energy

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
                atom_names = ATOM_MAP_14[self.seq[i]]
            else:
                atom_names = hy.ATOM_MAP_H[self.seq[i]]
            # Update atom names with terminal atoms
            if i == 0 and not heavy_only:
                atom_names = self.insert_terminal_atoms_into_name_list(
                    atom_names, ["H2", "H3"])
            elif i == len(self.seq) - 1 and not heavy_only:
                atom_names = self.insert_terminal_atoms_into_name_list(
                    atom_names, ["OXT"])
            all_atom_name_list.append(atom_names)

        if pprint:
            _size = self.atoms_per_res if not heavy_only else NUM_COORDS_PER_RES
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
                items = list(zip(flat_list, self.coords))
            elif self.has_hydrogens:
                items = list(zip(flat_list, self.hcoords))
            else:
                items = list(zip(flat_list, self.coords))
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

    def hydrogenrep_to_heavyatomrep(self, hcoords=None):
        """Remove hydrogens from a tensor of coordinates in heavy atom representation."""
        to_stack = []
        if hcoords is None:
            hcoords = self.hcoords

        for i, s in enumerate(self.seq3.split(" ")):
            num_heavy = 4 + len(SC_BUILD_INFO[s]['atom-names'])
            h_start = i * hy.NUM_COORDS_PER_RES_W_HYDROGENS
            h_end = h_start + num_heavy

            n_pad = NUM_COORDS_PER_RES - num_heavy

            to_stack.extend(
                [hcoords[h_start:h_end],
                 torch.zeros(n_pad, 3, requires_grad=True)])

        return torch.cat(to_stack)

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
        if isinstance(self.angles, torch.Tensor):
            self.angles = self.angles.cuda()
        else:
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
        if not torch.is_tensor(self.coords):
            self.coords = torch.tensor(self.coords)
        if not torch.is_tensor(self.hcoords):
            self.hcoords = torch.tensor(self.hcoords)
        if not torch.is_tensor(self.angles):
            self.angles = torch.tensor(self.angles)
        self.is_numpy = False


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
