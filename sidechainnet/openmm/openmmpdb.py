from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from pdbfixer import PDBFixer
import io
from simtk.openmm import LocalEnergyMinimizer
from collections import defaultdict, OrderedDict
import prody as pr
import numpy as np

import sidechainnet as scn


class OpenMMPDB(object):
    """Operates on a single PDB object in Sidechainnet Calculates energy, force norms,
    force per all atoms, force per atoms present in Sidechainnent."""

    def __init__(self, pdbstr, mask=None, pdbid='unknown'):
        self.pdbstr = pdbstr
        self.pdbid = pdbid
        self.pdb = PDBFixer(pdbfile=io.StringIO(pdbstr))
        self._pos_atom_map(self.pdb.positions)
        self.pdb.findMissingResidues()
        self.pdb.findMissingAtoms()
        self.pdb.findNonstandardResidues()
        self.pdb.addMissingAtoms()
        self.pdb.addMissingHydrogens(7.0)
        # self.pdb.replaceNonstandardResidues()
        self.chain = None
        self._set_up_env()
        self.mask = mask

    def _get_atom_residue(self):
        """Get Atom Name and Residue Name for Each Atom present in Sidechainnent
        PBDString."""
        for line in self.pdbstr.split('\n'):
            if 'ATOM' in line:
                spltline = list(filter(lambda a: a != '', line.split(' ')))
                yield spltline[2], spltline[3]

    def _pos_atom_map(self, init_positions):
        self.pos_atom_map = OrderedDict()
        for pos, atomres in zip(init_positions, self._get_atom_residue()):
            _pos = ''.join(["%.4f" % item for item in pos.value_in_unit(nanometer)])
            self.pos_atom_map[_pos] = atomres

    def _set_up_env(self):
        """This is called at __init__.

        Sets up the environment for energy and gradients. Returns Nothing.
        """
        self.forcefield = ForceField('amber14-all.xml', 'amber14/protein.ff14SB.xml')
        self.modeller = Modeller(self.pdb.topology, self.pdb.positions)
        self.system = self.forcefield.createSystem(self.modeller.topology,
                                                   nonbondedMethod=NoCutoff,
                                                   implicitSolvent=GBSAOBCForce)
        self.integrator = VerletIntegrator(1.0)
        self.context = Context(self.system, self.integrator)
        self.context.setPositions(self.modeller.positions)
        self.state = self.context.getState(getVelocities=True,
                                           getPositions=True,
                                           getParameters=True,
                                           getEnergy=True,
                                           getForces=True)

    def get_position(self):
        if self.state is None:
            self._set_up_env()
        return self.state.getPositions(
        )  # This returns the position in nanometers. In PDB File, the coordinates are in Amstrong

    def get_forces_per_all_atom(self):
        """Returns Force Gradients acting on all atoms including Hydrogens and Missing
        residue atoms added by PDBFixer."""

        if self.state is None:
            self._set_up_env()
        return self.state.getForces()

    def _pos_force_map(self):
        """Creates a Map between SideChainNet atoms and the Force Gradients acting on
        them.

        It filters out all the Hydrogens and missing residue atoms by PDBFixer
        """
        self.pos_force_map = dict()
        positions = self.state.getPositions()
        forces = self.state.getForces()
        for pos, force in zip(positions, forces):
            _pos = ''.join(["%.4f" % item for item in pos.value_in_unit(nanometer)])
            if _pos in self.pos_atom_map:
                self.pos_force_map[_pos] = force
        return self.pos_force_map

    def get_forces_per_atoms(self):
        """Returns Force Gradients acting on the atoms present in sidechainnet.

        It omits the force gradients on all the Hydrogens and missing residue
        atoms by PDBFixer.
        """
        if self.state is None:
            self._set_up_env()
        return self._pos_force_map()

    def get_potential_energy(self):
        """Returns potential energy of the state.

        Should be used as loss in forward pass.
        """
        if self.state is None:
            self._set_up_env()
        return self.state.getPotentialEnergy()

    def get_forcenorm(self):
        """Returns Force Norm for all Atoms (including Hydrogens and atoms added by
        PDBFixer)"""
        if self.state is None:
            self._set_up_env()
        forces = self.state.getForces()
        forceNormSum = 0.0 * kilojoules**2 / mole**2 / nanometer**2
        for f in forces:
            forceNormSum += dot(f, f)
        forceNorm = sqrt(forceNormSum)
        return forceNorm

    def localenergyminimize(self):
        LocalEnergyMinimizer.minimize(self.context, maxIterations=100)

    def minimize_energy(self, pin_gap_boundaries=True):
        """Perform an energy minimization simulation.

        Recommended AMBER forcefields from
        http://docs.openmm.org/latest/userguide/application.html.
        """
        modeller = Modeller(self.pdb.topology, self.pdb.positions)
        forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
        modeller.addHydrogens(forcefield)
        system = forcefield.createSystem(modeller.topology, nonbondedMethod=app.NoCutoff,
                nonbondedCutoff=1*nanometer, constraints=HBonds)
        if pin_gap_boundaries:
            system = self.pin_gap_boundaries(system)
        integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.004*picoseconds)
        simulation = Simulation(modeller.topology, system, integrator)
        simulation.context.setPositions(modeller.positions)
        simulation.minimizeEnergy()
        self.state = simulation.context.getState(getVelocities=True,
                                                 getPositions=True,
                                                 getParameters=True,
                                                 getEnergy=True,
                                                 getForces=True)

    def pin_gap_boundaries(self, system):
        """Pin residues on either side of internal gaps by setting masses to 0.

        Args:
            system (OpenMM System): System containing information about particles & mass.

        Returns:
            OpenMM System: A modified system with the residues immediately to the
            left/right of each gap immobilized.
        """
        boundaries = get_zeroindexed_gap_positions(self.mask)
        if not boundaries:
            return system
        new_system = set_residue_masses_to_zero(boundaries, self.pdb.topology, system)
        return new_system


    def make_high_and_low_energy_pdbs(self):
        """Minimize energy and make high/low energy PDB files."""
        self.to_pdb(f"{self.pdbid}_high")
        high_e = self.get_potential_energy()

        self.minimize_energy()
        self.to_pdb(f"{self.pdbid}_low")
        rmsd = scn.structure.structure.compare_pdb_files(f"{self.pdbid}_high.pdb",
                                                         f"{self.pdbid}_low.pdb")
        energy_d = high_e - self.get_potential_energy()
        print(f"Created files {self.pdbid}_{{high,low}}.pdb.")
        print(f"RMSD = {rmsd:.2f} A, Energy Delta = {energy_d.format('%.2f')}")

    def to_pdb(self, output_prefix):
        positions = self.get_position()
        with open(f'{output_prefix}.pdb', 'w') as f:
            PDBFile.writeFile(self.modeller.topology, positions, f)

    def get_coords(self):
        coords = []
        for v in self.get_position():
            coords.append(np.asarray(v.value_in_unit(angstrom)))
        coords = np.vstack(coords)
        return coords

    def get_atomnames(self):
        if self.chain is None:
            self.chain = next(self.pdb.topology.chains())
        atom_names = [str(an).split()[2][1:-1] for an in self.chain.atoms()]
        return atom_names

    def get_resnames(self):
        if self.chain is None:
            self.chain = next(self.pdb.topology.chains())
        resnames = [str(an).split()[-1][1:-2] for an in self.chain.atoms()]
        return resnames

    def get_resnums(self):
        if self.chain is None:
            self.chain = next(self.pdb.topology.chains())
        resnums = [int(str(an).split()[-2]) for an in self.chain.atoms()]
        return resnums

    def make_prody_atomgroup(self):
        ag = pr.AtomGroup()
        ag.setCoords(self.get_coords())
        ag.setNames(self.get_atomnames())
        ag.setResnames(self.get_resnames())
        ag.setResnums(self.get_resnums())
        return ag


def get_zeroindexed_gap_positions(mask, return_lengths=False):
    """Return [start-1, end+1] position of every gap in mask, zero-indexed.

    Args:
        mask (str): A SidechainNet mask (sequence of '+' and '-') marking sequence gaps.
        return_lengths (bool): If True, also return lengths of each interior gap.

    Returns:
        list: A list of indices describing the residues immediately preceding and
        following each internal gap in the mask. The numbers are indexed counting only
        the present residues. If there are not internal gaps, returns an empty list.

        If return_lengths == True, returns a Tuple whose second element is a list of the
        relevant gap lengths.

    Examples:
        >>> get_zeroindexed_gap_positions("+++--+")
        [2, 3]
        >>> get_zeroindexed_gap_positions("+++--+", True)
        ([2, 3], [2])
        >>> get_zeroindexed_gap_positions("+++-------+")
        [2, 3]
        >>> get_zeroindexed_gap_positions("---++++++---")
        []
    """
    gap_locs = []
    gap_lengths = []
    gap_len_counter = 0
    cur_pos = 0
    in_gap = False
    for m in mask:
        if cur_pos == 0 and m == "-":
            in_gap = True
        elif m == "+" and not in_gap:
            cur_pos += 1
        elif m == "-" and not in_gap:
            gap_locs.append(cur_pos - 1)
            in_gap = True
            gap_len_counter += 1
        elif m == "-" and in_gap:
            gap_len_counter += 1
            continue
        elif m == "+" and in_gap and cur_pos != 0:
            in_gap = False
            gap_locs.append(cur_pos)
            gap_lengths.append(gap_len_counter)
            gap_len_counter = 0
            cur_pos += 1
        elif m == "+" and in_gap and cur_pos == 0:
            in_gap = False
            cur_pos += 1
    if in_gap:
        gap_locs.pop()

    if return_lengths:
        return gap_locs, gap_lengths

    return gap_locs


def set_residue_masses_to_zero(res_indices, topo, sys):
    """Set the masses for the specified residues in a system and topology to zero.

    Args:
        res_indices (list): List of integers of residues that will be immobilized.
        topo (OpenMM Topology): Topology of PDB structure (PDBStructure.topology).
        sys (OpenMM System): System containing atomic masses.

    Returns:
        OpenMM System: Modified system with the masses for all atoms in specified residues
            set to zero.
    """
    residues = list(topo.residues())
    for i in res_indices:
        res = residues[i]
        atom_indices = [a.index for a in res.atoms()]
        for ai in atom_indices:
            sys.setParticleMass(ai, 0)
    return sys


if __name__ == "__main__":

    d = scn.load("debug", "/home/jok120/sidechainnet/data/sidechainnet/")
    sb = scn.StructureBuilder(d['valid-70']['seq'][13], d['valid-70']['crd'][13])

    fixer = PDBFixer(pdbfile=io.StringIO(sb.to_pdbstr()))
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.findNonstandardResidues()
