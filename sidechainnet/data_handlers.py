from sidechainnet.structure.build_info import NUM_COORDS_PER_RES
from sidechainnet import structure
import numpy as np
import prody
from sidechainnet.utils.sequence import ONE_TO_THREE_LETTER_MAP
from openmm.app import element as elem
from openmm.app.forcefield import ForceField, HBonds
from openmm.app.modeller import Modeller
from openmm.openmm import LangevinMiddleIntegrator
from sidechainnet.structure.PdbBuilder import ATOM_MAP_14
from openmm.app import Topology
from openmm import Vec3
from openmm.unit import nanometer, angstroms, kelvin, picosecond, picoseconds
import openmm

import sidechainnet
import sidechainnet.structure.hydrogens.hydrogens as hy
from sidechainnet.structure.structure import coord_generator


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

    def get_data_split(self, split_name):
        """Return list of proteins belonging to str split_name."""
        return [p for p in self if p.split == split_name]

    def __getitem__(self, id):
        """Retrieve protein by index or ID (name, i.e. '1A9U_1_A')."""
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
    """Represent a single protein in SidechainNet."""

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

    def __len__(self):
        """Return length of protein sequence."""
        return len(self.seq)

    def to_3Dmol(self):
        """Return an interactive visualization of the protein with py3DMol."""
        if self.sb is None:
            self.sb = sidechainnet.StructureBuilder(self.seq, self.coords)
        return self.sb.to_3Dmol()

    def to_pdb(self, path, title=None):
        """Save structure to path as a PDB file."""
        if not title:
            title = self.id
        if self.sb is None:
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

    def get_openmm_repr(self, skip_missing_residues=True):
        """Return tuple of OpenMM topology and positions for analysis with OpenMM."""
        self.positions = []
        self.topology = Topology()
        self.openmm_seq = ""
        chain = self.topology.addChain()
        coord_gen = coord_generator(self.coords, self.atoms_per_res)
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
            for an, c in zip(atom_names, coords):
                if an == "PAD":
                    continue
                self.topology.addAtom(name=an,
                                      element=get_element_from_atomname(an),
                                      residue=residue)
                pos = Vec3(c[0], c[1], c[2]) * angstroms
                self.positions.append(pos)
            self.openmm_seq += residue_code
        self.topology.createStandardBonds()
        self.topology.createDisulfideBonds(self.positions)
        return self.topology, self.positions

    def make_pdbfixer(self):
        """Construct and return an OpenMM PDBFixer object for this protein."""
        from pdbfixer import PDBFixer
        t, p = self.get_openmm_repr()
        self.pdbfixer = PDBFixer(topology=t,
                                 positions=p,
                                 sequence=self.openmm_seq,
                                 use_topology=True)
        return self.pdbfixer

    def run_pdbfixer(self):
        """Add missing atoms to protein's PDBFixer representation."""
        self.pdbfixer.findMissingResidues()
        self.pdbfixer.findMissingAtoms()
        # print("Missing atoms", self.pdbfixer.missingAtoms)
        self.pdbfixer.findNonstandardResidues()
        self.pdbfixer.addMissingAtoms()
        self.pdbfixer.addMissingHydrogens(7.0)

    def minimize_pdbfixer(self):
        """Perform an energy minimization using the PDBFixer representation. Return ∆E."""
        self.modeller = Modeller(self.pdbfixer.topology, self.pdbfixer.positions)
        self.forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
        self.system = self.forcefield.createSystem(self.modeller.topology,
                                                   nonbondedMethod=openmm.app.NoCutoff,
                                                   nonbondedCutoff=1 * nanometer,
                                                   constraints=HBonds)
        # self.modeller.addHydrogens(7.0)
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
        self.make_pdbfixer()
        self.run_pdbfixer()
        return self.minimize_pdbfixer()

    def get_rmsd_difference(self):
        """Report difference in start/end coordinates after energy minimization."""
        aligned_minimized = prody.calcTransformation(
            self.ending_positions, self.starting_positions).apply(self.ending_positions)
        rmsd = prody.calcRMSD(self.starting_positions, aligned_minimized)
        return rmsd

    def add_hydrogens(self):
        """Add hydrogens to the internal protein structure representation."""
        self.sb = structure.StructureBuilder(self.seq, crd=self.coords)
        self.sb.add_hydrogens()
        self.coords = self.sb.coords
        self.has_hydrogens = True
        self.atoms_per_res = hy.NUM_COORDS_PER_RES_W_HYDROGENS

    def __repr__(self) -> str:
        """Represent an SCNProtein as a string."""
        return (f"SCNProtein({self.id}, len={len(self)}, missing={self.num_missing}, "
                f"split='{self.split}')")


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
