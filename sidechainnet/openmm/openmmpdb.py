from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from pdbfixer import PDBFixer
import io
from simtk.openmm import LocalEnergyMinimizer
from collections import defaultdict

import sidechainnet as scn


class OpenMMPDB(object):
    """Operates on a single PDB object in Sidechainnet Calculates energy, force norms,
    force per all atoms, force per atoms present in Sidechainnent."""

    def __init__(self, pdbstr, pdbid):
        self.pdbstr = pdbstr
        self.pdbid = pdbid
        self.pdb = PDBFixer(pdbfile=io.StringIO(pdbstr))
        self._pos_atom_map(self.pdb.positions)
        self.pdb.findMissingResidues()
        self.pdb.findMissingAtoms()
        self.pdb.addMissingAtoms()
        self.pdb.addMissingHydrogens(7.0)
        self._set_up_env()

    def _get_atom_residue(self):
        """Get Atom Name and Residue Name for Each Atom present in Sidechainnent
        PBDString."""
        for line in self.pdbstr.split('\n'):
            if 'ATOM' in line:
                spltline = list(filter(lambda a: a != '', line.split(' ')))
                yield spltline[2], spltline[3]

    def _pos_atom_map(self, init_positions):
        self.pos_atom_map = defaultdict(lambda: defaultdict(dict))
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
            if self.pos_atom_map.get(_pos) is not None:
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

    def minimize_energy(self):
        """Perform an energy minimization simulation."""
        modeller = Modeller(self.pdb.topology, self.pdb.positions)
        forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
        modeller.addHydrogens(forcefield)
        system = forcefield.createSystem(modeller.topology, nonbondedMethod=app.NoCutoff,
                nonbondedCutoff=1*nanometer, constraints=HBonds)
        integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.004*picoseconds)
        simulation = Simulation(modeller.topology, system, integrator)
        simulation.context.setPositions(modeller.positions)
        simulation.minimizeEnergy()
        self.state = simulation.context.getState(getVelocities=True,
                                                 getPositions=True,
                                                 getParameters=True,
                                                 getEnergy=True,
                                                 getForces=True)

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
