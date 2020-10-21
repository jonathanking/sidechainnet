import numpy as np
import torch

from sidechainnet.utils.sequence import VOCAB
from sidechainnet.structure.build_info import SC_BUILD_INFO, BB_BUILD_INFO, NUM_COORDS_PER_RES, SC_ANGLES_START_POS
from sidechainnet.structure.structure import nerf


class StructureBuilder(object):
    """
    Given angles and protein sequence, reconstructs a single protein's structure.

    The hydroxyl-oxygen of terminal residues is not placed because this would
    mean that the number of coordinates per residue would not be constant, or
    cause other complications (i.e. what if the last atom of a structure is not
    really a terminal atom because it's tail is masked out?). It is simpler to
    ignore this atom for now.
    """

    def __init__(self, seq, ang=None, coords=None, device=torch.device("cpu")):
        """Initialize a StructureBuilder for a single protein.

        Args:
            seq: An integer tensor or a string of length L that represents the protein's
                amino acid sequence.
            ang: A float tensor (L X NUM_PREDICTED_ANGLES) that contains all of the 
                protein's interior angles.
            device: An optional torch device on which to build the structure.
        """
        # Validate input data
        if (ang is None and coords is None) or (ang is not None and coords is not None):
            raise ValueError(
                "You must provide exactly one of either coordinates or angles.")
        if ang is not None and np.any(np.all(ang == 0, axis=1)):
            missing_loc = np.where(np.all(ang == 0, axis=1))
            raise ValueError(f"Building atomic coordinates from angles is not supported "
                             f"for structures with missing residues. Missing residues = "
                             f"{list(missing_loc[0])}. Protein structures with missing "
                             "residues are only supported if built directly from "
                             "coordinates (also supported by StructureBuilder).")
        if coords is not None:
            self.coords = coords
            self.coord_type = "numpy" if type(coords) is np.ndarray else 'torch'
        else:
            self.coords = []
            self.coord_type = "numpy" if type(ang) is np.ndarray else 'torch'

        self.seq = seq
        self.ang = ang
        self.device = device

        self.prev_ang = None
        self.prev_bb = None
        self.next_bb = None

        self.pdb_creator = None
        self.integer_coded_seq = np.asarray([VOCAB._char2int[s] for s in seq])

    def __len__(self):
        return len(self.seq)

    def _iter_resname_angs(self, start=0):
        for resname, angles in zip(self.integer_coded_seq[start:], self.ang[start:]):
            yield resname, angles

    def _build_first_two_residues(self):
        """ Constructs the first two residues of the protein. """
        resname_ang_iter = self._iter_resname_angs()
        first_resname, first_ang = next(resname_ang_iter)
        second_resname, second_ang = next(resname_ang_iter)
        first_res = ResidueBuilder(first_resname, first_ang, prev_res=None, next_res=None)
        second_res = ResidueBuilder(second_resname,
                                    second_ang,
                                    prev_res=first_res,
                                    next_res=None)

        # After building both backbones use the second residue's N to build the first's CB
        first_res.build_bb()
        second_res.build()
        first_res.next_res = second_res
        first_res.build_sc()

        return first_res, second_res

    def build(self):
        """
        Construct all of the atoms for a residue. Special care must be taken
        for the first residue in the sequence in order to place its CB, if
        present.
        """
        # If a StructureBuilder does not have angles, build returns its coordinates
        if self.ang is None:
            return self.coords

        # Build the first and second residues, a special case
        first, second = self._build_first_two_residues()

        # Combine the coordinates and build the rest of the protein
        self.coords = first.stack_coords() + second.stack_coords()

        # Build the rest of the structure
        prev_res = second
        for i, (resname, ang) in enumerate(self._iter_resname_angs(start=2)):
            res = ResidueBuilder(resname,
                                 ang,
                                 prev_res=prev_res,
                                 next_res=None,
                                 is_last_res=i + 2 == len(self.seq) - 1)
            self.coords += res.build()
            prev_res = res

        if self.coord_type == 'torch':
            self.coords = torch.stack(self.coords)
        else:
            self.coords = np.stack(self.coords)

        return self.coords

    def _initialize_coordinates_and_PdbCreator(self):
        if len(self.coords) == 0:
            self.build()

        if not self.pdb_creator:
            from sidechainnet.structure.PdbBuilder import PdbBuilder
            if self.coord_type == 'numpy':
                self.pdb_creator = PdbBuilder(self.seq, self.coords)
            else:
                self.pdb_creator = PdbBuilder(self.seq, self.coords.numpy())

    def to_pdb(self, path, title="pred"):
        self._initialize_coordinates_and_PdbCreator()
        self.pdb_creator.save_pdb(path, title)

    def to_gltf(self, path, title="pred"):
        self._initialize_coordinates_and_PdbCreator()
        self.pdb_creator.save_gltf(path, title)

    def to_3Dmol(self, style=None, **kwargs):
        import py3Dmol
        if not style:
            style = {'cartoon': {'color': 'spectrum'}, 'stick': {'radius': .15}}
        self._initialize_coordinates_and_PdbCreator()

        view = py3Dmol.view(**kwargs)
        view.addModel(self.pdb_creator.get_pdb_string(), 'pdb')
        if style:
            view.setStyle(style)
        view.zoomTo()
        return view


class ResidueBuilder(object):

    def __init__(self,
                 name,
                 angles,
                 prev_res,
                 next_res,
                 is_last_res=False,
                 device=torch.device("cpu")):
        """Initialize a residue builder for building a residue's coordinates from angles.  
        
        If prev_{bb, ang} are None, then this is the first residue. 

        Args:
            name: The integer amino acid code for this residue.
            angles: A float tensor containing necessary angles to define this residue.
            prev_bb: Tensor, None
            Coordinate tensor (3 x 3) of previous residue, upon which this residue is extending.
            prev_ang : Tensor, None
            Angle tensor (1 X NUM_PREDICTED_ANGLES) of previous reside, upon which this residue is extending.
        """
        if type(name) != np.int64 and type(name) != torch.Tensor:
            raise ValueError("Expected integer AA code." + str(name.shape) +
                             str(type(name)))
        if type(angles) == np.ndarray:
            angles = torch.tensor(angles, dtype=torch.float32)
        self.name = name
        self.ang = angles.squeeze()
        self.prev_res = prev_res
        self.next_res = next_res
        self.device = device
        self.is_last_res = is_last_res

        self.bb = []
        self.sc = []
        self.coords = []
        self.coordinate_padding = torch.zeros(3)

    def build(self):
        self.build_bb()
        self.build_sc()
        return self.stack_coords()

    def build_bb(self):
        """ Builds backbone for residue. """
        if self.prev_res is None:
            self.bb = self.init_bb()
        else:
            pts = [self.prev_res.bb[0], self.prev_res.bb[1], self.prev_res.bb[2]]
            for j in range(4):
                if j == 0:
                    # Placing N
                    t = self.prev_res.ang[4]  # thetas["ca-c-n"]
                    b = BB_BUILD_INFO["BONDLENS"]["c-n"]
                    dihedral = self.prev_res.ang[1]  # psi of previous residue
                elif j == 1:
                    # Placing Ca
                    t = self.prev_res.ang[5]  # thetas["c-n-ca"]
                    b = BB_BUILD_INFO["BONDLENS"]["n-ca"]
                    dihedral = self.prev_res.ang[2]  # omega of previous residue
                elif j == 2:
                    # Placing C
                    t = self.ang[3]  # thetas["n-ca-c"]
                    b = BB_BUILD_INFO["BONDLENS"]["ca-c"]
                    dihedral = self.ang[0]  # phi of current residue
                else:
                    # Placing O (carbonyl)
                    t = torch.tensor(BB_BUILD_INFO["BONDANGS"]["ca-c-o"])
                    b = BB_BUILD_INFO["BONDLENS"]["c-o"]
                    if self.is_last_res:
                        # we explicitly measure this angle during dataset creation,
                        # no need to invert it here.
                        dihedral = self.ang[1]
                    else:
                        # the angle for placing oxygen is opposite to psi of current res
                        dihedral = self.ang[1] - np.pi

                next_pt = nerf(pts[-3], pts[-2], pts[-1], b, t, dihedral)
                pts.append(next_pt)
            self.bb = pts[3:]

        return self.bb

    def init_bb(self):
        """ Initialize the first 3 points of the protein's backbone. Placed in an arbitrary plane (z = .001). """
        n = torch.tensor([0, 0, 0.001], device=self.device)
        ca = n + torch.tensor([BB_BUILD_INFO["BONDLENS"]["n-ca"], 0, 0],
                              device=self.device)
        cx = torch.cos(np.pi - self.ang[3]) * BB_BUILD_INFO["BONDLENS"]["ca-c"]
        cy = torch.sin(np.pi - self.ang[3]) * BB_BUILD_INFO["BONDLENS"]['ca-c']
        c = ca + torch.tensor([cx, cy, 0], device=self.device, dtype=torch.float32)
        o = nerf(n, ca, c, torch.tensor(BB_BUILD_INFO["BONDLENS"]["c-o"]),
                 torch.tensor(BB_BUILD_INFO["BONDANGS"]["ca-c-o"]),
                 self.ang[1] - np.pi)  # opposite to current residue's psi
        return [n, ca, c, o]

    def build_sc(self):
        """
        Builds the sidechain atoms for this residue.

        Care is taken when placing the first sc atom (the beta-Carbon). This is
        because the dihedral angle that places this atom must be defined using
        a neighboring (previous or next) residue.
        """
        assert len(self.bb) > 0, "Backbone must be built first."
        self.pts = {"N": self.bb[0], "CA": self.bb[1], "C": self.bb[2]}
        if self.next_res:
            self.pts["N+"] = self.next_res.bb[0]
        else:
            self.pts["C-"] = self.prev_res.bb[2]

        last_torsion = None
        for i, (bond_len, angle, torsion,
                atom_names) in enumerate(get_residue_build_iter(self.name,
                                                                SC_BUILD_INFO)):
            # Select appropriate 3 points to build from
            if self.next_res and i == 0:
                a, b, c = self.pts["N+"], self.pts["C"], self.pts["CA"]
            elif i == 0:
                a, b, c = self.pts["C-"], self.pts["N"], self.pts["CA"]
            else:
                a, b, c = (self.pts[an] for an in atom_names[:-1])

            # Select appropriate torsion angle, or infer it if it's part of a planar configuration
            if type(torsion) is str and torsion == "p":
                torsion = self.ang[SC_ANGLES_START_POS + i]
            elif type(torsion) is str and torsion == "i" and last_torsion:
                torsion = last_torsion - np.pi

            new_pt = nerf(a, b, c, bond_len, angle, torsion)
            self.pts[atom_names[-1]] = new_pt
            self.sc.append(new_pt)
            last_torsion = torsion

        return self.sc

    def stack_coords(self):
        self.coords = self.bb + self.sc + (NUM_COORDS_PER_RES - len(self.bb) -
                                           len(self.sc)) * [self.coordinate_padding]
        return self.coords

    def __repr__(self):
        return f"ResidueBuilder({VOCAB.int2char(int(self.name))})"


def get_residue_build_iter(res, build_dictionary):
    """
    For a given residue integer code and a residue building data dictionary,
    this function returns an iterator that returns 4-tuples. Each tuple
    contains the necessary information to generate the next atom in that
    residue's sidechain. This includes the bond lengths, bond angles, and
    torsional angles.
    """
    r = build_dictionary[VOCAB.int2chars(int(res))]
    bvals = [torch.tensor(b, dtype=torch.float32) for b in r["bonds-vals"]]
    avals = [torch.tensor(a, dtype=torch.float32) for a in r["angles-vals"]]
    tvals = [
        torch.tensor(t, dtype=torch.float32) if t not in ["p", "i"] else t
        for t in r["torsion-vals"]
    ]
    return iter(zip(bvals, avals, tvals, [t.split("-") for t in r["torsion-names"]]))


if __name__ == '__main__':
    import pickle

    def load_data(path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data

    d = load_data(
        "/home/jok120/dev_sidechainnet/data/sidechainnet/sidechainnet_casp12_30.pkl")

    i = 15

    sb = StructureBuilder(d['train']['seq'][i], d['train']['ang'][i])
    sb.to_pdb("test00.pdb")
