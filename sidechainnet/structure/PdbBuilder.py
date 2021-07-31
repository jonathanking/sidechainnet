"""A class for creating PDB files/strings given a protein's sequence and coordinates."""
import itertools

import numpy as np
from sidechainnet.structure.build_info import NUM_COORDS_PER_RES, SC_BUILD_INFO
from sidechainnet.structure.HydrogenBuilder import ATOM_MAP_24, NUM_COORDS_PER_RES_W_HYDROGENS
from sidechainnet.structure.structure import coord_generator
from sidechainnet.utils.sequence import ONE_TO_THREE_LETTER_MAP


class PdbBuilder(object):
    """Creates a PDB file given a protein's atomic coordinates and sequence.

    The general idea is that if any model is capable of predicting a set of coordinates
    and mapping between those coordinates and residue/atom names, then this object can
    be use to transform that output into a PDB file.

    The Python format string was taken from http://cupnet.net/pdb-format/.
    """

    def __init__(self,
                 seq,
                 coords,
                 atoms_per_res=NUM_COORDS_PER_RES,
                 terminal_atoms=None):
        """Initialize a PdbBuilder.

        Args:
            coords: A numpy matrix of shape (L x N) x 3, where L is the protein sequence
                length and N is the number of atoms per residue in the coordinate set.
            seq: A length L string representing the protein sequence with  one character
                per amino acid.
            atoms_per_res: The number of atoms recorded per residue. This must be the
                same for every residue.
            terminal_atoms: Python dictionary mapping "H2", "H3", and "OXT" atom names to
                their positions in the structure.
        """
        if len(seq) != coords.shape[0] / atoms_per_res:
            raise ValueError(
                "The sequence length must match the coordinate length and contain 1 "
                "letter AA codes." + str(coords.shape[0] / atoms_per_res) + " " +
                str(len(seq)))
        if coords.shape[0] % atoms_per_res != 0:
            raise AssertionError(f"Coords is not divisible by {atoms_per_res}. "
                                 f"{coords.shape}")
        if atoms_per_res not in (NUM_COORDS_PER_RES, NUM_COORDS_PER_RES_W_HYDROGENS):
            raise ValueError(
                f"Values for atoms_per_res other than {NUM_COORDS_PER_RES}"
                f"/{NUM_COORDS_PER_RES_W_HYDROGENS} are currently not supported.")

        self.atoms_per_res = atoms_per_res
        self.has_hydrogens = self.atoms_per_res == NUM_COORDS_PER_RES_W_HYDROGENS
        self.coords = coords
        self.seq = seq
        self.mapping = self._make_mapping_from_seq()
        self.terminal_atoms = terminal_atoms

        # PDB Formatting Information
        self.format_str = ("{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}"
                           "{:8.3f}{:6.2f}{:6.2f}          {:>2s}{:2s}")
        self.defaults = {
            "alt_loc": "",
            "chain_id": "A",
            "insertion_code": "",
            "occupancy": 1,
            "temp_factor": 0,
            "element_sym": "",
            "charge": ""
        }
        self.title = "Untitled"
        self.atom_nbr = 1
        self.res_nbr = 1
        self._pdb_str = ""
        self._pdb_body_lines = []
        self._pdb_lines = []

    def _coord_generator(self):
        """Return a generator to iteratively yield self.atoms_per_res atoms at a time."""
        return coord_generator(self.coords, self.atoms_per_res)

    def _get_line_for_atom(self, res_name, atom_name, atom_coords, missing=False):
        """Return the 'ATOM...' line in PDB format for the specified atom.

        If missing, this function should have special, but not yet determined,
        behavior.
        """
        if missing:
            occupancy = 0
        else:
            occupancy = self.defaults["occupancy"]
        return self.format_str.format(
            "ATOM", self.atom_nbr, atom_name, self.defaults["alt_loc"],
            ONE_TO_THREE_LETTER_MAP[res_name], self.defaults["chain_id"], self.res_nbr,
            self.defaults["insertion_code"], atom_coords[0], atom_coords[1],
            atom_coords[2], occupancy, self.defaults["temp_factor"], atom_name[0],
            self.defaults["charge"])

    def _get_lines_for_residue(self,
                               res_name,
                               atom_names,
                               coords,
                               n_terminal=False,
                               c_terminal=False):
        """Return a list of PDB-formatted lines for all atoms in a single residue.

        Calls get_line_for_atom.
        """
        residue_lines = []
        for atom_name, atom_coord in zip(atom_names, coords):
            if (atom_name == "PAD" or np.isnan(atom_coord).sum() > 0 or
                    atom_coord.sum() == 0):
                continue
            residue_lines.append(self._get_line_for_atom(res_name, atom_name, atom_coord))
            self.atom_nbr += 1

        # Add Terminal Atoms (Must be provided in terminal_atoms dict; Hs must be built)
        if n_terminal and self.terminal_atoms:
            residue_lines.append(
                self._get_line_for_atom(res_name, "H2", self.terminal_atoms["H2"]))
            residue_lines.append(
                self._get_line_for_atom(res_name, "H3", self.terminal_atoms["H3"]))
            self.atom_nbr += 2
        if c_terminal and self.terminal_atoms:
            residue_lines.append(
                self._get_line_for_atom(res_name, "OXT", self.terminal_atoms["OXT"]))

        return residue_lines

    def _get_lines_for_protein(self):
        """Return a list of PDB-formatted lines for all residues in this protein.

        Calls get_lines_for_residue.
        """
        self._pdb_body_lines = []
        self.res_nbr = 1
        self.atom_nbr = 1
        mapping_coords = zip(self.mapping, self._coord_generator())
        for index, ((res_name, atom_names), res_coords) in enumerate(mapping_coords):
            # TODO assumes only first/last residue have terminal atoms
            self._pdb_body_lines.extend(
                self._get_lines_for_residue(res_name,
                                            atom_names,
                                            res_coords,
                                            n_terminal=index == 0,
                                            c_terminal=index == len(self.seq) - 1))
            self.res_nbr += 1
        return self._pdb_body_lines

    def _make_header(self, title):
        """Return a string representing the PDB header."""
        return f"REMARK  {title}" + "\n" + self._make_SEQRES()

    @staticmethod
    def _make_footer():
        """Return a string representing the PDB footer."""
        return "TER\nEND          \n"

    def _make_mapping_from_seq(self):
        """Given a protein sequence, this returns a mapping that assumes coords are
        generated in groups of atoms_per_res (the output is L x atoms_per_res x 3.)"""
        atom_names = ATOM_MAP_14 if not self.has_hydrogens else ATOM_MAP_24
        mapping = []
        for residue in self.seq:
            mapping.append((residue, atom_names[residue]))
        return mapping

    def get_pdb_string(self, title=None):
        if not title:
            title = self.title

        if self._pdb_str:
            return self._pdb_str
        self._get_lines_for_protein()
        self._pdb_lines = [self._make_header(title)
                           ] + self._pdb_body_lines + [self._make_footer()]
        self._pdb_str = "\n".join(self._pdb_lines)
        return self._pdb_str

    def _make_SEQRES(self):
        """Return a SEQRES entry as a multi-line string for this PDB file."""
        three_letter_seq = [ONE_TO_THREE_LETTER_MAP[c] for c in self.seq]
        residue_blocks = list(split_every(13, three_letter_seq))
        lineno = 1
        nres = len(self.seq)
        lines = []
        for block in residue_blocks:
            res_block_str = " ".join(block)
            cur_line = f"SEQRES {lineno: >3} {self.defaults['chain_id']} {nres: >4}  {res_block_str: <61}"
            lines.append(cur_line)
            lineno += 1
        return "\n".join(lines)

    def save_pdb(self, path, title="UntitledProtein"):
        """Write out the generated PDB file as a string to the specified path."""
        with open(path, "w") as outfile:
            outfile.write(self.get_pdb_string(title))

    def save_gltf(self, path, title="test", create_pdb=False):
        """First creates a PDB file, then converts it to GLTF and saves it to disk.

        Used for visualizing with Weights and Biases.
        """
        import pymol
        assert ".gltf" in path, "requested filepath must end with '.gtlf'."
        if create_pdb:
            self.save_pdb(path.replace(".gltf", ".pdb"), title)
        pymol.cmd.load(path.replace(".gltf", ".pdb"), title)
        pymol.cmd.color("oxygen", title)
        pymol.cmd.save(path, quiet=True)
        pymol.cmd.delete("all")


def split_every(n, iterable):
    """Split iterable into chunks. From https://stackoverflow.com/a/1915307/2780645."""
    i = iter(iterable)
    piece = list(itertools.islice(i, n))
    while piece:
        yield piece
        piece = list(itertools.islice(i, n))


ATOM_MAP_14 = {}
for one_letter in ONE_TO_THREE_LETTER_MAP.keys():
    ATOM_MAP_14[one_letter] = ["N", "CA", "C", "O"] + list(
        SC_BUILD_INFO[ONE_TO_THREE_LETTER_MAP[one_letter]]["atom-names"])
    ATOM_MAP_14[one_letter].extend(["PAD"] * (14 - len(ATOM_MAP_14[one_letter])))
