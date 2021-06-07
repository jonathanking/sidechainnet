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


class SCNDataset(object):

    def __init__(self, data) -> None:
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
        return [p for p in self if p.split == split_name]

    def __getitem__(self, id):
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
        return len(self.idx_to_SCNProtein)

    def __iter__(self):
        yield from self.ids_to_SCNProtein.values()

    def __repr__(self) -> str:
        return f"SCNDataset(n={len(self)})"

    def filter_ids(self, to_keep):
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

    def __len__(self):
        return len(self.seq)

    def to_3Dmol(self):
        if self.sb is None:
            self.sb = sidechainnet.StructureBuilder(self.seq, self.coords)
        return self.sb.to_3Dmol()

    def to_pdb(self):
        if self.sb is None:
            self.sb = sidechainnet.StructureBuilder(self.seq, self.coords)
        return self.sb.to_pdb()

    @property
    def missing(self):
        return self.mask.count("-")

    @property
    def seq3(self):
        return " ".join([ONE_TO_THREE_LETTER_MAP[c] for c in self.seq])

    def get_openmm_repr(self, skip_missing_residues=True):
        self.positions = []
        self.topology = Topology()
        self.openmm_seq = ""
        chain = self.topology.addChain()
        coord_generator = _coord_generator(self.coords)
        for i, (residue_code, coords,
                mask_char) in enumerate(zip(self.seq, coord_generator, self.mask)):
            residue_name = ONE_TO_THREE_LETTER_MAP[residue_code]
            if mask_char == "-" and skip_missing_residues:
                continue
            atom_names = ATOM_MAP_14[residue_code]
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
        from pdbfixer import PDBFixer
        t, p = self.get_openmm_repr()
        self.pdbfixer = PDBFixer(topology=t,
                                 positions=p,
                                 sequence=self.openmm_seq,
                                 use_topology=True)
        return self.pdbfixer

    def run_pdbfixer(self):
        self.pdbfixer.findMissingResidues()
        self.pdbfixer.findMissingAtoms()
        # print("Missing atoms", self.pdbfixer.missingAtoms)
        self.pdbfixer.findNonstandardResidues()
        self.pdbfixer.addMissingAtoms()
        self.pdbfixer.addMissingHydrogens(7.0)

    def minimize_pdbfixer(self):
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
        self.make_pdbfixer()
        self.run_pdbfixer()
        return self.minimize_pdbfixer()

    def get_rmsd_difference(self):
        aligned_minimized = prody.calcTransformation(
            self.ending_positions, self.starting_positions).apply(self.ending_positions)
        rmsd = prody.calcRMSD(self.starting_positions, aligned_minimized)
        return rmsd

    def __repr__(self) -> str:
        return f"SCNProtein({self.id}, len={len(self)}, missing={self.mask.count('-')}, split='{self.split}')"


def _coord_generator(coords, atoms_per_res=14):
    """Return a generator to iteratively yield atoms_per_res atoms at a time."""
    coord_idx = 0
    while coord_idx < coords.shape[0]:
        yield coords[coord_idx:coord_idx + atoms_per_res]
        coord_idx += atoms_per_res


def get_element_from_atomname(atom_name):
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
    else:
        raise ValueError(f"Unknown element for atom name {atom_name}.")
