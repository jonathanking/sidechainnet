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
        for i, (residue_code, coords, mask_char, atom_names) in enumerate(
                zip(self.seq, coord_gen, self.mask, self.get_atom_names())):
            residue_name = ONE_TO_THREE_LETTER_MAP[residue_code]
            if mask_char == "-" and skip_missing_residues:
                continue
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
                if an == "PAD":
                    continue
                self.topology.addAtom(name=an,
                                      element=get_element_from_atomname(an),
                                      residue=residue)
                self.positions.append(self._Vec3(c))

                # If this is a heavy atom from atommap14, record (coord14idx->openmmmidx)
                if not an.startswith("H"):
                    self.atommap14idx_to_openmmidx[coordidx] = (an, openmmidx)
                    coordidx += 1

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
        for i, (residue_code, coords, mask_char, atom_names) in enumerate(
                zip(self.seq, coord_gen, self.mask, self.get_atom_names())):
            if mask_char == "-" and skip_missing_residues:
                continue
            for j, (an, c) in enumerate(zip(atom_names, coords)):
                if an == "PAD":
                    continue
                self.positions.append(self._Vec3(c))

        return self.positions

    def insert_terminal_atoms_into_name_list(self, atom_names, terminal_atom_names):
        """Insert a list of atoms into an existing list, keeping lengths the same."""
        pad_idx = atom_names.index("PAD")
        new_list = list(atom_names)
        new_list[pad_idx:pad_idx + len(terminal_atom_names)] = terminal_atom_names
        return new_list

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

    def get_energy(self):
        """Return potential energy of the system given current atom positions."""
        if not self.openmm_initialized:
            self.initialize_openmm()
        self.simulation.context.setPositions(self.positions)
        self.starting_state = self.simulation.context.getState(getEnergy=True,
                                                               getForces=True)
        self.starting_energy = self.starting_state.getPotentialEnergy()
        return self.starting_energy

    def get_forces_dict(self):
        """Return a dictionary mapping resnum -> {atomname: (resname, force)}."""
        if not self.starting_energy:
            self.get_energy()
        forces = self.starting_state.getForces(
        )  # Matches OpenMM Topology; includes terminal atoms and Hs
        force_dict = {}
        for atom, force in zip(self.topology.atoms(), forces):
            resnum = atom.residue.index
            atomname = atom.name
            resname = atom.residue.name
            if resnum not in force_dict:
                force_dict[resnum] = {}
            force_dict[resnum][atomname] = resname, force / (kilojoule_per_mole /
                                                             angstrom)
        return force_dict

    def get_forces(self, output_rep="heavy", sum_hydrogens=False, pprint=False):
        """Return tensor of forces as requested."""
        if output_rep == "heavy":
            outdim = NUM_COORDS_PER_RES  # 14
        elif output_rep == "all":
            outdim = hy.NUM_COORDS_PER_RES_W_HYDROGENS  # 24
        else:
            raise ValueError(f"{output_rep} is not a valid output representation. "
                             "Choose either 'heavy' or 'all'.")
        if sum_hydrogens and output_rep == "all":
            raise ValueError("Cannot sum hydrogen forces for output dim 'all'.")

        force_dict = self.get_forces_dict()
        force_array = torch.zeros(len(self.seq) * outdim, 3, dtype=torch.float64)
        force_array_raw = torch.zeros(len(self.seq) * outdim, 3, dtype=torch.float64)
        force_idx = 0

        for resnum in range(len(force_dict)):
            cur_atom_num = 0
            atom_positions = {}
            for atomname, (resname, f) in force_dict[resnum].items():
                if (atomname.startswith("H") and output_rep == "heavy" and
                        not sum_hydrogens):
                    break
                elif atomname.startswith("H") and output_rep == "heavy" and sum_hydrogens:
                    heavyatom = hy.HYDROGEN_NAMES_TO_PARTNERS[resname][atomname]
                    heavyatom_idx = atom_positions[heavyatom]
                    force_array[heavyatom_idx] += torch.tensor([f.x, f.y, f.z],
                                                               dtype=torch.float64)
                    continue
                force_array[force_idx] = force_array_raw[force_idx] = torch.tensor(
                    [f.x, f.y, f.z], dtype=torch.float64)
                atom_positions[atomname] = force_idx
                force_idx += 1
                cur_atom_num += 1

            force_idx += (outdim - cur_atom_num)  # skip padded atoms

        if pprint:
            atom_name_pprint(self.get_atom_names(heavy_only=output_rep=="heavy"), force_array)
            return

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

    def update_hydrogens(self, hcoords):
        """Take a set a hydrogen coordinates and use it to update this protein."""
        mask = self.get_hydrogen_coord_mask()
        self.hcoords = hcoords * mask
        self.has_hydrogens = True
        self.atoms_per_res = hy.NUM_COORDS_PER_RES_W_HYDROGENS
        self.update_positions()

    def get_hydrogen_coord_mask(self):
        """Return a torch tensor with 0s representing pad characters in self.hcoords."""
        mask = torch.zeros_like(self.hcoords)
        for i, (res3, res1) in enumerate(zip(self.seq3.split(" "), self.seq)):
            n_heavy_atoms = sum(
                [True if an != "PAD" else False for an in ATOM_MAP_14[res1]])
            n_hydrogens = len(hy.HYDROGEN_NAMES[res3])
            mask[i *
                 hy.NUM_COORDS_PER_RES_W_HYDROGENS:i * hy.NUM_COORDS_PER_RES_W_HYDROGENS +
                 n_heavy_atoms + n_hydrogens, :] = 1.0
        return mask

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

            to_stack.extend(
                [hcoords[h_start:h_end],
                 torch.zeros(n_pad, 3, requires_grad=True)])

        return torch.cat(to_stack)

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
            print(" ".join([f"{n: <3}" for n in nums]))
            for ans in all_atom_name_list:
                ans = [f"{a: <3}" for a in ans]
                print(" ".join(ans))
            return None

        # Prints a representation of coordinates and their atom names
        if zip_coords:
            flat_list = [item for sublist in all_atom_name_list for item in sublist]
            if heavy_only:
                items = list(zip(self.coords, flat_list))
            elif self.has_hydrogens:
                items = list(zip(self.hcoords, flat_list))
            else:
                items = list(zip(self.coords, flat_list))
            for i in items:
                i = list(i)
                i[0] = i[0].detach().numpy()
                print(i)
            return None
        return all_atom_name_list


def atom_name_pprint(atom_names, values):
    """Nicely print atom names and values."""
    flat_list = [item for sublist in atom_names for item in sublist]
    for i, (an, vals) in enumerate(zip(flat_list, values)):
        print(f"{i: <2}", f"{an: <3}", vals)


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
