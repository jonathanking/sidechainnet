"""Defines high-level objects for interfacing with raw SidechainNet data.

To utilize SCNDataset, pass scn_dataset=True to scn.load().

    >>> d = scn.load("debug", scn_dataset=True)
    >>> d
    SCNDataset(n=461)

SCNProteins may be iterated over or selected from the SCNDataset.

    >>> d["1HD1_1_A"]
    SCNProtein(1HD1_1_A, len=75, missing=0, split='train')

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
from sidechainnet.utils.measure import GLOBAL_PAD_CHAR
from openmm import Vec3, Platform
from openmm.app import Topology
from openmm.app import element as elem
from openmm.app.forcefield import ForceField, HBonds
from openmm.app.modeller import Modeller
from openmm.openmm import LangevinMiddleIntegrator
from openmm.unit import angstroms, kelvin, nanometer, picosecond, picoseconds, kilojoule_per_mole, angstrom

import sidechainnet
import sidechainnet.structure.HydrogenBuilder as hy
from sidechainnet import structure
from sidechainnet.structure.build_info import NUM_COORDS_PER_RES, SC_BUILD_INFO
from sidechainnet.structure.PdbBuilder import ATOM_MAP_14
from sidechainnet.structure.structure import coord_generator
from sidechainnet.utils.sequence import ONE_TO_THREE_LETTER_MAP

OPENMM_FORCEFIELDS = ['amber14/protein.ff15ipq.xml', 'amber14/spce.xml']


class SCNDataset(object):
    """A representation of a SidechainNet dataset."""

    def __init__(self, data) -> None:
        """Initialize a SCNDataset from underlying SidechainNet formatted dictionary."""
        super().__init__()
        # Determine available datasplits
        self.splits = []
        for split_name in ['train', 'valid', 'test']:
            for k in data.keys():
                if split_name in k:
                    self.splits.append(k)

        self.split_to_ids = {}
        self.ids_to_SCNProtein = {}
        self.idx_to_SCNProtein = {}

        # Create SCNProtein objects and add to data structure
        idx = 0
        for split in self.splits:
            d = data[split]
            for c, a, s, u, m, e, n, r, z, i in zip(d['crd'], d['ang'], d['seq'],
                                                    d['ums'], d['msk'], d['evo'],
                                                    d['sec'], d['res'], d['mod'],
                                                    d['ids']):
                try:
                    self.split_to_ids[split].append(i)
                except KeyError:
                    self.split_to_ids[split] = [i]

                p = SCNProtein(coordinates=c,
                               angles=a,
                               sequence=s,
                               unmodified_seq=u,
                               mask=m,
                               evolutionary=e,
                               secondary_structure=n,
                               resolution=r,
                               is_modified=z,
                               id=i,
                               split=split)
                self.ids_to_SCNProtein[i] = p
                self.idx_to_SCNProtein[idx] = p
                idx += 1

    def get_protein_list_by_split_name(self, split_name):
        """Return list of SCNProtein objects belonging to str split_name."""
        return [p for p in self if p.split == split_name]

    def __getitem__(self, id):
        """Retrieve a protein by index or ID (name, e.g. '1A9U_1_A')."""
        if isinstance(id, str):
            return self.ids_to_SCNProtein[id]
        elif isinstance(id, slice):
            step = 1 if not id.step else id.step
            stop = len(self) if not id.stop else id.stop
            start = 0 if not id.start else id.start
            stop = len(self) + stop if stop < 0 else stop
            start = len(self) + start if start < 0 else start
            return [self.idx_to_SCNProtein[i] for i in range(start, stop, step)]
        else:
            return self.idx_to_SCNProtein[id]

    def __len__(self):
        """Return number of proteins in the dataset."""
        return len(self.idx_to_SCNProtein)

    def __iter__(self):
        """Iterate over SCNProtein objects."""
        yield from self.ids_to_SCNProtein.values()

    def __repr__(self) -> str:
        """Represent SCNDataset as a string."""
        return f"SCNDataset(n={len(self)})"

    def filter_ids(self, to_keep):
        """Remove proteins whose IDs are not included in list to_keep."""
        to_delete = []
        for pnid in self.ids_to_SCNProtein.keys():
            if pnid not in to_keep:
                to_delete.append(pnid)
        for pnid in to_delete:
            p = self.ids_to_SCNProtein[pnid]
            self.split_to_ids[p.split].remove(pnid)
            del self.ids_to_SCNProtein[pnid]
        self.idx_to_SCNProtein = {}
        for i, protein in enumerate(self):
            self.idx_to_SCNProtein[i] = protein


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

    def __len__(self):
        """Return length of protein sequence."""
        return len(self.seq)

    def to_3Dmol(self):
        """Return an interactive visualization of the protein with py3DMol."""
        if self.sb is None:
            if self._has_hydrogens:
                self.sb = sidechainnet.StructureBuilder(self.seq, self.hcoords)
            else:
                self.sb = sidechainnet.StructureBuilder(self.seq, self.coords)
        return self.sb.to_3Dmol()

    def to_pdb(self, path, title=None):
        """Save structure to path as a PDB file."""
        if not title:
            title = self.id
        if self.sb is None:
            if self._has_hydrogens:
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

    def _Vec3(self, single_coord):
        return Vec3(single_coord[0], single_coord[1], single_coord[2]) * angstroms

    def get_openmm_repr(self, skip_missing_residues=True):
        """Return tuple of OpenMM topology and positions for analysis with OpenMM."""
        openmmidx = coordidx = coordidx24 = 0
        self.atommap14idx_to_openmmidx = {}
        self.atommap24idx_to_openmmidx = {}
        self.positions = []
        self.topology = Topology()
        self.openmm_seq = ""
        chain = self.topology.addChain()
        coord_gen = coord_generator(self.hcoords, self.atoms_per_res)
        placed_terminal_h = False
        placed_terminal_oxt = False
        for i, (residue_code, coords,
                mask_char) in enumerate(zip(self.seq, coord_gen, self.mask)):
            residue_name = ONE_TO_THREE_LETTER_MAP[residue_code]
            if mask_char == "-" and skip_missing_residues:
                continue
            # At this point, hydrogens should already be added
            if not self.has_hydrogens:
                atom_names = ATOM_MAP_14[residue_code]
            else:
                atom_names = hy.ATOM_MAP_24[residue_code]
            residue = self.topology.addResidue(name=residue_name, chain=chain)
            # Rectify coordinate 14 mapping index to account for padded atoms
            if i > 0:
                coordidx -= 1
                prev_residue_n_atoms = coordidx % NUM_COORDS_PER_RES
                if prev_residue_n_atoms != 0:
                    coordidx += NUM_COORDS_PER_RES - prev_residue_n_atoms

                coordidx24 -= 1
                prev_residue_n_atoms = coordidx24 % hy.NUM_COORDS_PER_RES_W_HYDROGENS
                if prev_residue_n_atoms != 0:
                    coordidx24 += hy.NUM_COORDS_PER_RES_W_HYDROGENS - prev_residue_n_atoms
            for j, (an, c) in enumerate(zip(atom_names, coords)):
                # Handle N-terminal Hydrogens
                if an == "PAD" and i == 0 and not placed_terminal_h:
                    self.topology.addAtom(name="H2",
                                          element=get_element_from_atomname("H2"),
                                          residue=residue)
                    self.positions.append(self._Vec3(self.sb.terminal_atoms["H2"]))
                    openmmidx += 1
                    self.topology.addAtom(name="H3",
                                          element=get_element_from_atomname("H3"),
                                          residue=residue)
                    self.positions.append(self._Vec3(self.sb.terminal_atoms["H3"]))
                    openmmidx += 1
                    placed_terminal_h = True
                    continue
                # Handle C-terminal OXT
                elif an == "PAD" and i == len(self.seq) - 1 and not placed_terminal_oxt:
                    self.topology.addAtom(name="OXT",
                                          element=get_element_from_atomname("OXT"),
                                          residue=residue)
                    self.positions.append(self._Vec3(self.sb.terminal_atoms["OXT"]))
                    openmmidx += 1
                    placed_terminal_oxt = True
                    continue
                elif an == "PAD":
                    continue
                self.topology.addAtom(name=an,
                                      element=get_element_from_atomname(an),
                                      residue=residue)
                self.positions.append(self._Vec3(c))

                # If this is a heavy atom from atommap14, record (coord14idx->openmmmidx)
                if not an.startswith("H"):
                    self.atommap14idx_to_openmmidx[coordidx] = (an, openmmidx)
                    coordidx += 1
                if an not in ["H2", "H3", "OXT"]:
                    self.atommap24idx_to_openmmidx[coordidx24] = (an, openmmidx)
                    coordidx24 += 1
                openmmidx += 1

            self.openmm_seq += residue_code
        self.topology.createStandardBonds()
        # TODO think about disulfide bonds at a later point, see CYS/CYX (bridge, no H)
        # self.topology.createDisulfideBonds(self.positions)
        return self.topology, self.positions

    def update_positions(self, skip_missing_residues=True):
        """Update the positions variable with hydrogen coordinate values."""
        self.positions = []
        coord_gen = coord_generator(self.hcoords, self.atoms_per_res)
        placed_terminal_h = False
        placed_terminal_oxt = False
        for i, (residue_code, coords,
                mask_char) in enumerate(zip(self.seq, coord_gen, self.mask)):
            if mask_char == "-" and skip_missing_residues:
                continue
            # At this point, hydrogens should already be added
            if not self.has_hydrogens:
                atom_names = ATOM_MAP_14[residue_code]
            else:
                atom_names = hy.ATOM_MAP_24[residue_code]
            for j, (an, c) in enumerate(zip(atom_names, coords)):
                # Handle N-terminal Hydrogens
                if an == "PAD" and i == 0 and not placed_terminal_h:
                    self.positions.append(self._Vec3(self.sb.terminal_atoms["H2"]))
                    self.positions.append(self._Vec3(self.sb.terminal_atoms["H3"]))
                    placed_terminal_h = True
                    continue
                # Handle C-terminal OXT
                elif an == "PAD" and i == len(self.seq) - 1 and not placed_terminal_oxt:
                    self.positions.append(self._Vec3(self.sb.terminal_atoms["OXT"]))
                    placed_terminal_oxt = True
                    continue
                elif an == "PAD":
                    continue
                self.positions.append(self._Vec3(c))

        return self.positions

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
        self.simulation = openmm.app.Simulation(self.modeller.topology, self.system,
                                                self.integrator, platform=self.platform,
                                                platformProperties=properties)
        self.openmm_initialized = True

    def get_energy(self):
        """Return potential energy of the system given current atom positions."""
        if not self.openmm_initialized:
            self.initialize_openmm()
        self.simulation.context.setPositions(self.positions)
        self.starting_state = self.simulation.context.getState(getEnergy=True,
                                                               getForces=True)
        self.starting_energy = self.starting_state.getPotentialEnergy()
        return self.starting_energy

    def get_forces(self):
        """Return an array of forces matching the heavy atom coordinate representation."""
        if not self.starting_energy:
            self.get_energy()
        forces = self.starting_state.getForces()  # Matches OpenMM Topology; includes terminal atoms and Hs
        force_array = torch.zeros(len(self.seq) * NUM_COORDS_PER_RES, 3, dtype=torch.float64)  # Matches 14-atom coords
        for atommap14_idx, (name, openmmidx) in self.atommap14idx_to_openmmidx.items():
            f = forces[openmmidx] / (kilojoule_per_mole / angstrom)
            force_array[atommap14_idx] = torch.tensor([f.x, f.y, f.z], dtype=torch.float64)
            # force_array[atommap14_idx] = torch.tensor([
            #     forces[openmmidx]._value[0], forces[openmmidx]._value[1],
            #     forces[openmmidx]._value[2]
            # ])
        return force_array

    def get_forces_w_hydrogens(self):
        """Return an array of forces matching the all atom coordinate representation."""
        if not self.starting_energy:
            self.get_energy()
        forces = self.starting_state.getForces(
        )  # Matches OpenMM Topology; includes terminal atoms and Hs
        force_array = torch.zeros(len(self.seq) * hy.NUM_COORDS_PER_RES_W_HYDROGENS,
                                  3,
                                  dtype=torch.float64)  # Matches 24-atom coords
        for atommap24_idx, (name, openmmidx) in self.atommap24idx_to_openmmidx.items():
            f = forces[openmmidx] / (kilojoule_per_mole / angstrom)
            force_array[atommap24_idx] = torch.tensor([f.x, f.y, f.z],
                                                      dtype=torch.float64)
            # force_array[atommap14_idx] = torch.tensor([
            #     forces[openmmidx]._value[0], forces[openmmidx]._value[1],
            #     forces[openmmidx]._value[2]
            # ])
        return force_array

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

    def add_hydrogens(self, from_angles=False, coords=None):
        """Add hydrogens to the internal protein structure representation."""
        if from_angles:
            self.sb = structure.StructureBuilder(self.seq, ang=self.angles)
            self.sb.build()
        elif coords is not None:
            self.sb = structure.StructureBuilder(self.seq, crd=coords)
        else:
            self.sb = structure.StructureBuilder(self.seq, crd=self.coords)
        self.sb.add_hydrogens()
        self.hcoords = self.sb.coords
        self.has_hydrogens = True
        self.atoms_per_res = hy.NUM_COORDS_PER_RES_W_HYDROGENS
        self.update_positions()

    def __repr__(self) -> str:
        """Represent an SCNProtein as a string."""
        return (f"SCNProtein({self.id}, len={len(self)}, missing={self.num_missing}, "
                f"split='{self.split}')")

    def hydrogenrep_to_heavyatomrep(self, hcoords):
        """Remove hydrogens from a tensor of coordinates in heavy atom representation."""
        to_stack = []

        for i, s in enumerate(self.seq3.split(" ")):
            num_heavy = 4 + len(SC_BUILD_INFO[s]['atom-names'])
            h_start = i * hy.NUM_COORDS_PER_RES_W_HYDROGENS
            h_end = h_start + num_heavy

            n_pad = NUM_COORDS_PER_RES - num_heavy

            to_stack.extend([hcoords[h_start:h_end], torch.zeros(n_pad, 3, requires_grad=True)])

        return torch.cat(to_stack)


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
