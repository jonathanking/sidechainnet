import re

#  This script helps produce several dictionaries that will aid in sidechain structure generation.
#  Run this as part of the setup for sidechains.py.

res_info = {
    "ARG": {
        "atom_types": ["n ", "cx", "c8", "c8", "C8", "N2", "ca", "n2"],
        "atom_names": ["n ", "CA", "CB", "CG", "CD", "NE", "CZ", "NH1"]
    },
    "HIS": {
        "atom_types": ["n ", "cx", "Ct", "cc", "cv"],
        "atom_names": ["n ", "CA", "CB", "ND1"]
    },
    "LYS": {
        "atom_types": ["n ", "cx", "c8", "c8", "C8", "C8", "N3"],
        "atom_names": ["n ", "CA", "CB", "CG", "CD", "CE", "NZ"]
    },
    "ASP": {
        "atom_types": ["n ", "cx", "2c", "Co", "O2"],
        "atom_names": ["n ", "CA", "CB", "CG", "OD1"]
    },
    "GLU": {
        "atom_types": ["n ", "cx", "2c", "2c", "co", "O2"],
        "atom_names": ["n ", "CA", "CB", "CG", "CD", "OE1"]
    },
    "SER": {
        "atom_types": ["n ", "cx", "2c", "Oh"],
        "atom_names": ["n ", "CA", "CB", "OG"]
    },
    "THR": {
        "atom_types": ["n ", "cx", "3c", "ct"],
        "atom_names": ["n ", "CA", "CB", "CG2"]
    },
    "ASN": {
        "atom_types": ["n ", "cx", "2c", "C ", "o "],
        "atom_names": ["n ", "CA", "CB", "CG", "ND2"]
    },
    "GLN": {
        "atom_types": ["n ", "cx", "2c", "2c", "c ", "o "],
        "atom_names": ["n ", "CA", "CB", "CG", "CD", "NE2"]
    },
    "CYS": {
        "atom_types": ["n ", "cx", "2c", "sh"],
        "atom_names": ["n ", "CA", "CB", "SG"]
    },
    "ALA": {
        "atom_types": ["n ", "cx", "ct"],
        "atom_names": ["n ", "CA", "CB"]
    },
    "VAL": {
        "atom_types": ["n ", "cx", "3c", "ct"],
        "atom_names": ["n ", "CA", "CB", "CG1"]
    },
    "ILE": {
        "atom_types": ["n ", "cx", "3c", "2c", "ct"],
        "atom_names": ["n ", "CA", "CB", "CG1", "CD1"]
    },
    "LEU": {
        "atom_types": ["n ", "cx", "2c", "3c", "ct"],
        "atom_names": ["n ", "CA", "CB", "CG", "CD1"]
    },
    "MET": {
        "atom_types": ["n ", "cx", "2c", "2c", "s ", "ct"],
        "atom_names": ["n ", "CA", "CB", "CG", "SD", "CE"]
    },
    "PHE": {
        "atom_types": ["n ", "cx", "2c", "ca", "ca"],
        "atom_names": ["n ", "CA", "CB", "CG", "CD1"]
    },
    "TYR": {
        "atom_types": ["n ", "cx", "2c", "ca", "ca"],
        "atom_names": ["n ", "CA", "CB", "CG", "CD1"]
    },
    "TRP": {
        "atom_types": ["n ", "cx", "ct", "c*", "cw"],
        "atom_names": ["n ", "CA", "CB", "CG", "CD1"]
    }
}

bond_angle_res_dict = {}

bonds = set()
angles = set()
bonds_both = set()
angles_both = set()

print("SC_DATA = {", end="")

sorted_res3 = list(sorted(res_info.keys()))

# for res, info in res_info.items():
for res in sorted_res3:
    info = res_info[res]
    print("\"{0}\": {{\"angles\": [".format(res), end="")
    types = info["atom_types"]
    bond_angle_res_dict[res] = {"bonds": [], "angles": []}
    for i in range(0, len(types) - 2):
        a = "{0}-{1}-{2}".format(types[i].lower(), types[i + 1].lower(),
                                 types[i + 2].lower())
        bond_angle_res_dict[res]["angles"].append(a)
        if i == len(types) - 3:
            print("\"{0}\"".format(a), end="")
        else:
            print("\"{0}\",".format(a), end="")
        angles.add(a)
        a2 = "{2}-{1}-{0}".format(types[i].lower(), types[i + 1].lower(),
                                  types[i + 2].lower())
        angles_both.add((a, a2))
    print("],\n\"bonds\": [", end="")
    for i in range(1, len(types) - 1):
        b = "{0}-{1}".format(types[i].lower(), types[i + 1].lower())
        bond_angle_res_dict[res]["bonds"].append(b)
        if i == len(types) - 2:
            print("\"{0}\"".format(b), end="")
        else:
            print("\"{0}\",".format(b), end="")
        bonds.add(b)
        b2 = "{1}-{0}".format(types[i].lower(), types[i + 1].lower())
        bonds_both.add((b, b2))
    print("]},")
print("}")
print("*" * 70)
print(bond_angle_res_dict)
print("*" * 70)
base = "/home/jok120/seq2struct/meta/"
with open(base+"forcefields/frcmod.ff14SB", "r") as f, open(base+"forcefields/gaff2.dat", "r") as f2, \
        open(base+"forcefields/parm99.dat", "r") as f3, open(base+"forcefields/parm10.dat", "r") as f4:
    forcefield = f.read()
    forcefield += f2.read()
    forcefield += f3.read()
    forcefield += f4.read()

bonds_both = sorted(list(bonds_both))
angles_both = sorted(list(angles_both))
bond_dict = {}
angle_dict = {}

for b1b2 in bonds_both:
    found_any = False
    for b in b1b2:
        if "*" in b:
            b = b.replace("*", r"\*")
        p = re.compile(r"[^-]" + b + r"[^-].*", re.IGNORECASE)
        r = p.findall(forcefield)
        if len(r) > 0:
            found_any = True
            print(b, p.findall(forcefield))
            break
    if not found_any:
        print(b, [])
        bond_dict[b] = None
    else:
        # bond_dict[b] = float(r[0].split()[2])
        bond_dict[b1b2[0]] = float(re.split(r'\s{2,}', r[0])[2])
        bond_dict[b1b2[1]] = float(re.split(r'\s{2,}', r[0])[2])

for b1b2 in angles_both:
    found_any = False
    for b in b1b2:
        if "*" in b:
            b = b.replace("*", r"\*")
        # p = re.compile(r"[^-]" + b + r"[^-h].*", re.IGNORECASE)
        p = re.compile(r"[^-]" + b + r"[^-].*", re.IGNORECASE)
        r = p.findall(forcefield)
        if len(r) > 0:
            found_any = True
            print(b, p.findall(forcefield))
            break
    if not found_any:
        print(b, [])
        angle_dict[b] = None
    else:
        # angle_dict[b] = float(r[0].split()[2])
        angle_dict[b1b2[0]] = float(re.split(r'\s{2,}', r[0])[2])
        angle_dict[b1b2[1]] = float(re.split(r'\s{2,}', r[0])[2])

print("*" * 70)
print("BONDLENS =", bond_dict)
print("BONDANGS =", angle_dict)

print("*" * 70)

# for b in angles_both:
#     if "*" in b:
#         b = b.replace("*", r"\*")
#     p = re.compile(r"[^-]"+b+r"[^-].*", re.IGNORECASE)
#     print(b, p.findall(forcefield))

print("Bonds:")
print(sorted(list(bonds)))
print("**********")
print("Angles:")
print(sorted(list(angles)))
