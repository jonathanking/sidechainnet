"""A class for generating/visualizing protein atomic coordinates from measured angles."""

import copy
from io import UnsupportedOperation
import numpy as np
import prody as pr
import torch
from sidechainnet.structure.build_info import GLOBAL_PAD_CHAR
from sidechainnet.utils.measure import compute_fictious_atom_for_res1

from sidechainnet.utils.sequence import ONE_TO_THREE_LETTER_MAP, VOCAB
from sidechainnet.structure.build_info import SC_HBUILD_INFO, BB_BUILD_INFO, NUM_COORDS_PER_RES, SC_ANGLES_START_POS, NUM_ANGLES
from sidechainnet.structure.structure import nerf
from sidechainnet.structure.HydrogenBuilder import HydrogenBuilder, NUM_COORDS_PER_RES_W_HYDROGENS


class StructureBuilder(object):
    """Reconstruct a protein's structure given its sequence and angles or coordinates.

    The hydroxyl-oxygen of terminal residues is not placed because this would
    mean that the number of coordinates per residue would not be constant, or
    cause other complications (i.e. what if the last atom of a structure is not
    really a terminal atom because it's tail is masked out?).
    """

    def __init__(self,
                 seq,
                 ang=None,
                 crd=None,
                 device='cpu',
                 nerf_method="standard",
                 has_hydrogens=None):
        """Initialize a StructureBuilder for a single protein. Does not build coordinates.

        To generate coordinates after initialization, see build().
        To create PDB/GLTF files or to generate a py3Dmol visualization, see
        to_{pdb,gltf,3Dmol}.

        Args:
            seq: An integer tensor or a string of length L that represents the protein's
                amino acid sequence.
            ang: A float tensor (L X NUM_PREDICTED_ANGLES) that contains all of the
                protein's interior angles.
            crd: A float tensor ((L X NUM_COORDS_PER_RES) X 3) that contains all of the
                protein's atomic coordinates. Each residue must contain the same number
                of coordinates, with empty coordinate entries padded with 0-vectors.
            device: An optional torch device on which to build the structure.
            nerf_method (str, optional): Which NeRF implementation to use. "standard" uses
                the standard NeRF formulation described in many papers. "sn_nerf" uses an
                optimized version with less vector normalizations. Defaults to
                "standard".
            has_hydrogens(bool, optional): True if the coordinate matrix uses a hydrogen
                representation. If not provided, attempts to infer.
        """
        # TODO support one-hot sequences
        # Perhaps the user mistakenly passed coordinates for the angle arguments
        if ang is not None and crd is None and ang.shape[-1] == 3:
            self.coords = ang
            self.ang = None
        elif crd is not None and ang is None:
            self.ang = None
            self.coords = crd
            if len(self.coords.shape) == 3:
                raise ValueError("Batches of structures are not supported by "
                                 "StructureBuilder. See BatchedStructureBuilder instead.")
        elif crd is None and ang is not None:
            self.coords = None
            self.ang = ang
            if len(self.ang.shape) == 3:
                raise ValueError("Batches of structures are not supported by "
                                 "StructureBuilder. See BatchedStructureBuilder instead.")
        elif (ang is None and crd is None) or (ang is not None and crd is not None):
            raise ValueError("You must provide exactly one of either coordinates (crd) "
                             "or angles (ang).")

        self.seq_as_str = seq if type(seq) == str else _convert_seq_to_str(seq)
        self.seq_as_ints = np.asarray([VOCAB._char2int[s] for s in self.seq_as_str])
        self.device = device

        # Validate input data
        if self.coords is not None:
            self.is_numpy = False if isinstance(self.coords, torch.Tensor) else True
        else:
            self.is_numpy = False if isinstance(self.ang, torch.Tensor) else True
        self.array_lib = np if self.is_numpy else torch

        if self.ang is not None and self.ang.shape[-1] != NUM_ANGLES:
            raise ValueError(f"Angle matrix dimensions must match (L x {NUM_ANGLES}). "
                             f"You have provided {tuple(self.ang.shape)}.")
        if (self.coords is not None and self.coords.shape[-1] != 3):
            raise ValueError(f"Coordinate matrix dimensions must match (L x 3). "
                             f"You have provided {tuple(self.coords.shape)}.")
        if (self.coords is not None and (self.coords.shape[0]) != len(self.seq_as_str) and
            (self.coords.shape[0]) != len(self.seq_as_str)):
            raise ValueError(
                f"The length of the coordinate matrix must match the sequence length. "
                f"You have provided coords.shape[0] = {self.coords.shape[0]}.")
        if self.ang is not None and (self.array_lib.isnan(self.ang)).all(axis=1).any():
            missing_loc = self.array_lib.where(
                (self.array_lib.isnan(self.ang)).all(axis=1))
            raise ValueError(f"Building atomic coordinates from angles is not supported "
                             f"for structures with missing residues. Missing residues = "
                             f"{list(missing_loc[0])}. Protein structures with missing "
                             "residues are only supported if built directly from "
                             "coordinates (also supported by StructureBuilder).")

        self.prev_ang = None
        self.prev_bb = None
        self.next_bb = None
        self.pdb_creator = None
        self.nerf_method = nerf_method
        if has_hydrogens is not None:
            self.has_hydrogens = has_hydrogens
        else:
            try:
                assert not (
                    self.coords.shape[0] % NUM_COORDS_PER_RES == 0 and
                    self.coords.shape[0] % NUM_COORDS_PER_RES_W_HYDROGENS == 0
                ), ("Coordinate tensor for protein has an ambiguous shape. Please pass a "
                    "value to has_hydrogens for clarification.")
                self.has_hydrogens = self.coords.shape[
                    0] % NUM_COORDS_PER_RES_W_HYDROGENS == 0
            except AttributeError:
                self.has_hydrogens = False
        self.terminal_atoms = None

    def __len__(self):
        """Return length of the protein sequence.

        Returns:
            int: Integer sequence length.
        """
        return len(self.seq_as_str)

    def _iter_resname_angs(self, start=0):
        for resname, angles in zip(self.seq_as_ints[start:], self.ang[start:]):
            yield resname, angles

    def _build_first_two_residues(self):
        """Construct the first two residues of the protein."""
        resname_ang_iter = self._iter_resname_angs()
        first_resname, first_ang = next(resname_ang_iter)
        second_resname, second_ang = next(resname_ang_iter)
        first_res = ResidueBuilder(first_resname,
                                   first_ang,
                                   prev_res=None,
                                   nerf_method=self.nerf_method,
                                   device=self.device)
        second_res = ResidueBuilder(second_resname,
                                    second_ang,
                                    prev_res=first_res,
                                    nerf_method=self.nerf_method,
                                    device=self.device)

        first_res.build()
        second_res.build()

        return first_res, second_res

    def build(self):
        """Construct all of the atoms for a residue.

        Special care must be taken for the first residue in the sequence in
        order to place its CB, if present.

        Returns:
            (numpy.ndarray, torch.Tensor): An array or tensor of the generated coordinates
            with shape ((L X NUM_COORDS_PER_RES) X 3).
        """
        # If a StructureBuilder does not have angles, build returns its coordinates
        if self.ang is None:
            return self.coords

        # Build the first and second residues, a special case
        first, second = self._build_first_two_residues()

        # Combine the coordinates and build the rest of the protein
        self.coords = first._stack_coords() + second._stack_coords()

        # Build the rest of the structure
        prev_res = second
        for i, (resname, ang) in enumerate(self._iter_resname_angs(start=2)):
            res = ResidueBuilder(resname,
                                 ang,
                                 prev_res=prev_res,
                                 is_last_res=i + 2 == len(self.seq_as_str) - 1,
                                 nerf_method=self.nerf_method,
                                 device=self.device)
            self.coords += res.build()
            prev_res = res

        if self.data_type == 'numpy' and torch.is_tensor(self.coords[0]):
            self.coords = torch.stack(self.coords)
            self.coords = self.coords.detach().numpy()
        elif self.data_type == 'numpy' and isinstance(self.coords[0], np.ndarray):
            self.coords = np.stack(self.coords)
        else:
            self.coords = torch.stack(self.coords)
            self.data_type = 'torch'

        return self.coords

    def _initialize_coordinates_and_PdbCreator(self):
        if self.coords is None or len(self.coords) == 0:
            self.build()

        if not self.pdb_creator:
            from sidechainnet.structure.PdbBuilder import PdbBuilder
            if self.is_numpy:
                self.pdb_creator = PdbBuilder(self.seq_as_str,
                                              self.coords,
                                              terminal_atoms=self.terminal_atoms,
                                              has_hydrogens=self.has_hydrogens)
            else:
                self.pdb_creator = PdbBuilder(self.seq_as_str,
                                              self.coords.cpu().detach().numpy(),
                                              terminal_atoms=self.terminal_atoms,
                                              has_hydrogens=self.has_hydrogens)

    def add_hydrogens(self):
        """Add Hydrogen atom coordinates to coordinate representation (re-apply PADs)."""
        if self.coords is None or not len(self.coords):
            raise ValueError("Cannot add hydrogens to a structure whose heavy atoms have"
                             " not yet been built.")
        self.hb = HydrogenBuilder(self.seq_as_str, self.coords, self.device)
        self.coords = self.hb.build_hydrogens()
        self.has_hydrogens = True
        self.terminal_atoms = self.hb.terminal_atoms

    def to_pdb(self, path, title="pred"):
        """Save protein structure as a PDB file to given path.

        Args:
            path (str): Path to save PDB file.
            title (str, optional): Title of structure for PDB file. Defaults to "pred".
        """
        self._initialize_coordinates_and_PdbCreator()
        self.pdb_creator.save_pdb(path, title)

    def to_pdbstr(self, title="pred"):
        """Return protein structure as a PDB string.

        Args:
            title (str, optional): Title of structure for PDB file. Defaults to "pred".
        """
        self._initialize_coordinates_and_PdbCreator()
        return self.pdb_creator.get_pdb_string(title)

    def to_gltf(self, path, title="pred"):
        """Save protein structure as a GLTF (3D-object) file to given path.

        Args:
            path (str): Path to save GLTF file.
            title (str, optional): Title of structure for GLTF file. Defaults to "pred".
        """
        self._initialize_coordinates_and_PdbCreator()
        self.pdb_creator.save_gltf(path, title)

    def to_png(self, path):
        """Save protein structure as PNG, showing sidechains if available. Requires pdb.

        Args:
            path (str): Path to save file. 
        """
        import pymol
        assert ".png" in path, "requested filepath must end with '.png'."
        pymol.cmd.load(path.replace(".png", ".pdb"))
        pymol.cmd.select("sidechain")
        pymol.cmd.show("lines")
        pymol.cmd.png(path, width=800, height=800, quiet=0, dpi=200, ray=0)
        pymol.cmd.delete("all")

    def to_3Dmol(self, style=None, other_protein=None, **kwargs):
        """Generate protein structure & return interactive py3Dmol.view for visualization.

        Args:
            style (str, optional): Style string to be passed to py3Dmol for
                visualization. Defaults to None.

        Returns:
            py3Dmol.view object: A view object that is interactive in iPython notebook
                settings.
        """
        import py3Dmol

        view = py3Dmol.view(**kwargs)
        view.addModel(self.to_pdbstr(), 'pdb', {'keepH': True})

        # If we have another protein to compare, align the other protein and add to viz
        if other_protein is not None:
            # Create copies and nan-masks of coordinate data
            other_protein.numpy()
            other_protein_copy = other_protein.copy()
            # Remove rows with nans
            other_protein_copy_coords = other_protein_copy.coords.reshape(-1, 3)
            other_protein_copy_coords_nonans = other_protein_copy_coords[
                ~np.isnan(other_protein_copy_coords).any(axis=-1)]
            if torch.is_tensor(self.coords):
                self.coords = self.coords.detach().cpu().numpy()
            coords_copy = copy.deepcopy(self.coords).reshape(-1, 3)
            coords_copy_nonans = coords_copy[~np.isnan(coords_copy).any(axis=-1)]
            # Perform alignment between non-nan coords
            t = pr.calcTransformation(other_protein_copy_coords_nonans,
                                      coords_copy_nonans)
            aligned_coords = t.apply(other_protein_copy_coords)
            # Update coords in other protein
            other_protein_copy.coords = aligned_coords.reshape(len(other_protein_copy),
                                                               -1, 3)
            if other_protein_copy.has_hydrogens:
                other_protein_copy.hcoords = t.apply(
                    other_protein_copy.hcoords.reshape(-1, 3)).reshape(
                        len(other_protein_copy), -1, 3)
                other_protein_copy.coords = other_protein_copy.hcoords
            # Add viz to model
            other_protein_copy.sb = None
            view.addModel(other_protein_copy.to_pdbstr())

        # Set visualization style
        if not style:
            style = {'cartoon': {'color': 'spectrum'}, 'stick': {'radius': .15}}
        if other_protein is None:
            view.setStyle(style)
        elif other_protein is not None:
            style1 = {
                # 'cartoon': {
                #     'color': '#599BFB'
                # },
                'stick': {
                    'radius': .07,
                    'color': '#599BFB'  # Blue
                }
            }
            style2 = {
                # 'cartoon': {
                #     'color': '#FB5960'
                # },
                'stick': {
                    'radius': .15,
                    'color': '#FB5960'  # Red, other
                }
            }
            view.setStyle({"model": 0}, style1)
            view.setStyle({"model": 1}, style2)

        view.zoomTo()
        return view


class ResidueBuilder(object):
    """Builds the atomic coordinates from angles for a specified amino acid residue."""

    def __init__(self,
                 name,
                 angles,
                 prev_res,
                 is_last_res=False,
                 device=torch.device("cpu"),
                 nerf_method="standard"):
        """Initialize a residue builder for building a residue's coordinates from angles.

        If prev_{bb, ang} are None, then this is the first residue.

        Args:
            name: The integer amino acid code for this residue.
            angles: A float tensor containing necessary angles to define this residue.
            prev_bb: Coordinate tensor (3 x 3) of previous residue, upon which this
                residue is extending.
            prev_ang : Angle tensor (1 X NUM_PREDICTED_ANGLES) of previous reside, upon
                which this residue is extending.
            nerf_method (str, optional): Which NeRF implementation to use. "standard" uses
                the standard NeRF formulation described in many papers. "sn_nerf" uses an
                optimized version with less vector normalizations. Defaults to
                "standard".
        """
        if (not isinstance(name, np.int64) and not isinstance(name, np.int32) and
                not isinstance(name, int) and not isinstance(name, torch.Tensor)):
            raise ValueError("Expected integer AA code." + str(name.shape) +
                             str(type(name)))
        if isinstance(angles, np.ndarray):
            angles = torch.tensor(angles, dtype=torch.float32, device=device)
        self.name = name
        self.ang = angles.squeeze()
        self.prev_res = prev_res
        self.device = device
        self.is_last_res = is_last_res
        self.nerf_method = nerf_method

        self.bb = []
        self.sc = []
        self.coords = []
        self.coordinate_padding = torch.ones(3, requires_grad=True,
                                             device=self.device) * GLOBAL_PAD_CHAR

    @property
    def AA(self):
        """Return the one-letter amino acid code (str) for this residue."""
        return VOCAB.int2char(int(self.name))

    def build(self):
        """Construct and return atomic coordinates for this protein."""
        self.build_bb()
        self.build_sc()
        return self._stack_coords()

    def build_bb(self):
        """Build backbone for residue."""
        if self.prev_res is None:
            self.bb = self._init_bb()
        else:
            pts = [self.prev_res.bb[0], self.prev_res.bb[1], self.prev_res.bb[2]]
            for j in range(4):
                if j == 0:
                    # Placing N
                    t = self.prev_res.ang[4]  # thetas["ca-c-n"]
                    b = BB_BUILD_INFO["BONDLENS"]["c-n"]
                    pb = BB_BUILD_INFO["BONDLENS"]["ca-c"]  # pb is previous bond len
                    dihedral = self.prev_res.ang[1]  # psi of previous residue
                elif j == 1:
                    # Placing Ca
                    t = self.prev_res.ang[5]  # thetas["c-n-ca"]
                    b = BB_BUILD_INFO["BONDLENS"]["n-ca"]
                    pb = BB_BUILD_INFO["BONDLENS"]["c-n"]
                    dihedral = self.prev_res.ang[2]  # omega of previous residue
                elif j == 2:
                    # Placing C
                    t = self.ang[3]  # thetas["n-ca-c"]
                    b = BB_BUILD_INFO["BONDLENS"]["ca-c"]
                    pb = BB_BUILD_INFO["BONDLENS"]["n-ca"]
                    dihedral = self.ang[0]  # phi of current residue
                else:
                    # Placing O (carbonyl)
                    t = torch.tensor(BB_BUILD_INFO["BONDANGS"]["ca-c-o"],
                                     device=self.device)
                    b = BB_BUILD_INFO["BONDLENS"]["c-o"]
                    pb = BB_BUILD_INFO["BONDLENS"]["ca-c"]
                    if self.is_last_res:
                        # we explicitly measure this angle during dataset creation,
                        # no need to invert it here.
                        dihedral = self.ang[1]
                    else:
                        # the angle for placing oxygen is opposite to psi of current res
                        dihedral = self.ang[1] - np.pi
                next_pt = nerf(pts[-3],
                               pts[-2],
                               pts[-1],
                               torch.tensor(b, device=self.device),
                               t,
                               dihedral,
                               l_bc=torch.tensor(pb, device=self.device),
                               nerf_method=self.nerf_method)
                pts.append(next_pt)
            self.bb = pts[3:]

        return self.bb

    def _init_bb(self):
        """Initialize the first 3 points of the protein's backbone.

        Placed in an arbitrary plane (z = .001).
        """
        n = torch.tensor([0.0, 0.0, 0.001], device=self.device, requires_grad=True)
        ca = n + torch.tensor([BB_BUILD_INFO["BONDLENS"]["n-ca"], 0.0, 0.0],
                              device=self.device,
                              requires_grad=True)
        cx = torch.cos(np.pi - self.ang[3]) * BB_BUILD_INFO["BONDLENS"]["ca-c"]
        cy = torch.sin(np.pi - self.ang[3]) * BB_BUILD_INFO["BONDLENS"]['ca-c']
        c = ca + torch.tensor(
            [cx, cy, 0.0], device=self.device, dtype=torch.float32, requires_grad=True)
        o = nerf(
            n,
            ca,
            c,
            torch.tensor(BB_BUILD_INFO["BONDLENS"]["c-o"], device=self.device),
            torch.tensor(BB_BUILD_INFO["BONDANGS"]["ca-c-o"], device=self.device),
            self.ang[1] - np.pi,  # opposite to current residue's psi
            l_bc=torch.tensor(BB_BUILD_INFO["BONDLENS"]["ca-c"],
                              device=self.device),  # Previous bond length
            nerf_method=self.nerf_method)
        return [n, ca, c, o]

    def build_sc(self):
        """Build the sidechain atoms for this residue.

        Care is taken when placing the first sc atom (the beta-Carbon). This is
        because the dihedral angle that places this atom must be defined using a
        neighboring (previous or next) residue.
        """
        assert len(self.bb) > 0, "Backbone must be built first."
        self.atom_names = ["N", "CA", "C", "O"]
        self.pts = {"N": self.bb[0], "CA": self.bb[1], "C": self.bb[2]}
        if self.prev_res:
            self.pts["C-"] = self.prev_res.bb[2]
        else:
            self.pts["C-"] = compute_fictious_atom_for_res1(self.pts["N"], self.pts["CA"],
                                                            self.pts["C"])

        last_torsion = None
        for i, (bond_len, angle, torsion, atom_names) in enumerate(
                _get_residue_build_iter(self.name, SC_HBUILD_INFO, self.device)):
            # Select appropriate 3 points to build from
            if i == 0:
                a, b, c = self.pts["C-"], self.pts["N"], self.pts["CA"]
            else:
                a, b, c = (self.pts[an] for an in atom_names[:-1])

            # Select appropriate torsion angle, or infer if part of a planar configuration
            if type(torsion) is str and torsion == "p":
                torsion = self.ang[SC_ANGLES_START_POS + i]
            elif type(torsion) is str and torsion == "i" and last_torsion is not None:
                torsion = last_torsion - np.pi

            new_pt = nerf(a, b, c, bond_len, angle, torsion)
            self.pts[atom_names[-1]] = new_pt
            self.sc.append(new_pt)
            last_torsion = torsion
            self.atom_names.append(atom_names[-1])

        return self.sc

    def _stack_coords(self):
        self.coords = self.bb + self.sc + (NUM_COORDS_PER_RES - len(self.bb) -
                                           len(self.sc)) * [self.coordinate_padding]
        return self.coords

    def to_prody(self, res):
        import prody as pr
        ag = pr.AtomGroup()
        ag.setCoords(torch.stack(self.bb + self.sc).detach().numpy())
        ag.setNames(self.atom_names)
        ag.setResnames([ONE_TO_THREE_LETTER_MAP[VOCAB._int2char[self.name]]] *
                       len(self.atom_names))
        ag.setResnums([res.getResnum()] * len(self.atom_names))
        return pr.Residue(ag, [0] * len(self.atom_names), None)

    def __repr__(self):
        """Return a string describing the name of the residue used for this object."""
        return f"ResidueBuilder({self.AA})"


def _get_residue_build_iter(res, build_dictionary, device):
    """Return an iterator over (bond-lens, angles, torsions, atom names) for a residue.

    This function makes it easy to iterate over the huge amount of data contained in
    the dictionary sidechainnet.structure.build_info.SC_BUILD_INFO. This dictionary
    contains all of the various standard bond and angle values that are used during atomic
    reconstruction of a residue from its angles.

    Args:
        res (int): An interger representing the integer code for a particular amino acid,
            e.g. 'Ala' == 'A' == 0 in sequence.py.
        build_dictionary (dict): A dictionary mapping 3-letter amino acid codes to
            dictionaries of information relevant to the construction of this amino acid
            from angles (i.e. angle names, atom types, bond lengths, bond types, torsion
            types, etc.). See sidechainnet.structure.build_info.SC_BUILD_INFO.

    Returns:
        iterator: An iterator that yields 4-tuples of (bond-value, angle-value,
        torsion-value, atom-name). These values can be used to generating atomic
        coordinates for a residue via the NeRF algorithm
        (sidechainnet.structure.structure.nerf).
    """
    r = build_dictionary[VOCAB.int2chars(int(res))]
    bond_vals = []
    angle_vals = []
    torsion_vals = []
    atom_names = []

    for i in range(len(r['bond-vals'])):
        if r['torsion-names'][i].split("-")[-1].startswith("H"):
            # We have reached the end of the heavy atoms
            break
        bond_vals.append(
            torch.tensor(r['bond-vals'][i], dtype=torch.float32, device=device))
        angle_vals.append(
            torch.tensor(r['angle-vals'][i], dtype=torch.float32, device=device))
        torsion_vals.append(
            torch.tensor(r['torsion-vals'][i], dtype=torch.float32, device=device
                        ) if r['torsion-vals'][i] not in ["p", "i"] else r['torsion-vals'][i])
        atom_names.append(r["torsion-names"][i].split("-"))

    return iter(zip(bond_vals, angle_vals, torsion_vals, atom_names))


def _convert_seq_to_str(seq):
    """Assuming seq is an int list or int tensor, returns its str representation."""
    seq_as_str = ""
    if isinstance(seq, torch.Tensor):
        seq = seq.numpy()
    seq = seq.flatten()
    if len(seq.shape) == 1:
        # The seq is represented as an integer sequence
        if len(seq.shape) == 1:
            seq_as_str = VOCAB.ints2str(seq)
        elif len(seq.shape) == 2:
            if seq.shape[0] != 1:
                raise ValueError(f"Seq shape {seq.shape} is not supported.")
            else:
                seq_as_str = VOCAB.ints2str(seq[0])
    else:
        raise UnsupportedOperation(f"Seq shape {seq.shape} is not supported.")

    return seq_as_str


# @torch.jit.script
def _init_bb_helper(BB_n_ca: float, ang3: torch.Tensor, BB_ca_c: float, device: str):
    """Torchscript friendly helper for _init_bb. Currently unused."""
    n = torch.tensor([0.0, 0.0, 0.001], requires_grad=True, device=device)
    ca = n + torch.tensor([BB_n_ca, 0.0, 0.0], requires_grad=True, device=device)
    pi_minus_ang3 = np.pi - ang3
    c = ca + (torch.stack([
        torch.cos(np.pi - ang3),
        torch.sin(pi_minus_ang3),
        torch.tensor(0.0, device=device)
    ]) * BB_ca_c).float()
    return n, ca, c


if __name__ == '__main__':
    import pickle

    def _load_data(path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data

    d = _load_data(
        "/home/jok120/dev_sidechainnet/data/sidechainnet/sidechainnet_casp12_30.pkl")

    idx = 15

    sb = StructureBuilder(d['train']['seq'][idx], d['train']['ang'][idx])
    sb.to_pdb("test00.pdb")
