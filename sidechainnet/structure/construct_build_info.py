"""Given some minimally hand-constructed data, produce the complete build info dict."""
import copy
import re

import numpy as np

from sidechainnet.structure.build_info import RAW_BUILD_INFO


def _create_atomname_lookup():
    """Create a dictionary mapping resname3 to dict mapping atom_name to atom_type."""
    with open("/home/jok120/build/amber16/dat/leap/lib/amino12.lib", "r") as f:
        text = f.read()
    atom_name_lookup = {}
    amino_name_info = re.findall(
        r"!entry\.[A-Z]{3}\.unit\.atoms table[\s\S]+?(?=!entry\.[A-Z]{3})", text, re.S)
    for match in amino_name_info:
        lines = match.split("\n")
        resname = lines[0].split(".")[1]
        assert len(resname) != 4, "terminal res not supported"
        if resname == 'HID':
            resname = 'HIS'
        if resname not in RAW_BUILD_INFO:
            continue
        atom_name_lookup[resname] = {}
        for line in lines[1:]:
            if not line:
                continue
            atom_data = line.split()
            a1, a2 = atom_data[:2]
            a1 = a1.replace("\"", "")
            a2 = a2.replace("\"", "")
            atom_name_lookup[resname][a1] = a2
    return atom_name_lookup


class ForceFieldLookupHelper(object):
    """Reads in AMBER ForceField data to facilitate bond/angle lookups with Regex."""

    def __init__(self, *forcefield_files):
        """Ingest one or more forcefield files ordered in terms of priority."""
        self.ff_files = forcefield_files
        self.text = ""

        for fn in self.ff_files:
            with open(fn, "r") as f:
                self.text += f.read()

        # Here we make some pre-emptive corrections for values that are not adequate in
        # AMBER. AMBER seems to specify interior angles for histidine and proline wich are
        # much too large (120 degrees and 109.5 degrees, respectively). Here, I have
        # remeasured the offending values.

        self.overwrite_rules = {}
        histidine_interior_angles = ["CT-CC-NA", "CC-NA-CR", "NA-CR-NB", "CR-NB-CV"]
        for hia in histidine_interior_angles:
            self.overwrite_rules[('HIS', hia)] = np.deg2rad(108)
        self.overwrite_rules[('PRO', "N -CX-CT")] = np.deg2rad(101.8812)
        self.overwrite_rules[('PRO', "CX-CT-CT")] = np.deg2rad(103.6465)
        self.overwrite_rules[('PRO', "CX-CT-CT")] = np.deg2rad(103.3208)

    def get_value(self, resname, item, convert_to_rad=True):
        """Lookup the value (bond length or bond angle) for the item (ex 'CX-2C':1.526).

        Args:
            item (str): A string representing a bond (2 atoms) or angle (3 atoms) with
                atom types separated by hyphens.

        Returns:
            value (float): The value of item as specified in AMBER forcefield parameters.
        """
        assert len(item.split("-")) <= 3, "Does not support torsion lookup."

        if (resname, item) in self.overwrite_rules:
            return self.overwrite_rules[(resname, item)]

        def get_match(s):
            s = s.replace("*", "\\*")
            pattern = f"^{s}[^-].+"
            return re.search(pattern, self.text, re.MULTILINE)

        match = get_match(item)

        if match is None:
            # Second attempt - look for the reverse item
            item = "-".join(item.split("-")[::-1])
            match = get_match(item)
            if match is None:
                raise ValueError(f"No match found for {item}.")

        line = match.group(0)
        line = line.replace(item, "").strip()
        value = float(line.split()[1])
        if convert_to_rad and len(item.split("-")) == 3:
            return np.deg2rad(value)
        return value


def create_complete_hydrogen_build_param_dict():
    """Create a Python dictionary that describes all necessary info for building atoms."""
    # We need to record the following information: bond lengths, bond angles
    new_build_info = copy.deepcopy(RAW_BUILD_INFO)

    # First, we must lookup the atom type for each atom
    atom_type_map = _create_atomname_lookup()

    # We also make ready a way to lookup forcefield parameter values from raw files
    base = '/home/jok120/build/amber16/dat/leap/parm/'
    ff_helper = ForceFieldLookupHelper(base + "frcmod.ff14SB",
                                       base + "parm10.dat")

    # Now, let's generate the types of the bonds and angles we will need to lookup
    for resname, resinfo in RAW_BUILD_INFO.items():
        new_build_info[resname]['bond-types'] = []
        new_build_info[resname]['angle-types'] = []
        new_build_info[resname]['bond-vals'] = []
        new_build_info[resname]['angle-vals'] = []
        for torsion in resinfo['torsion-names']:
            atomtypes = [f"{atom_type_map[resname][an]:<2}" for an in torsion.split("-")]
            bond = "-".join(atomtypes[-2:])
            angle = "-".join(atomtypes[-3:])
            new_build_info[resname]['bond-types'].append(bond)
            new_build_info[resname]['angle-types'].append(angle)

            # Given names/types of necessary bonds/angles, we look up their AMBER vals
            new_build_info[resname]['bond-vals'].append(ff_helper.get_value(resname, bond))
            new_build_info[resname]['angle-vals'].append(ff_helper.get_value(resname, angle))

    return new_build_info


if __name__ == "__main__":
    import pprint
    import json
    hbp = create_complete_hydrogen_build_param_dict()
    hbp_str = pprint.pformat(hbp, indent=1, width=90, compact=False)
    with open("hbp.py", "w") as f:
        # f.write(json.dumps(hbp, indent=2))
        f.write(hbp_str)
