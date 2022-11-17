"""This script compares the energy of four structural models of the same protein:
    - The native structure from the PDB,                                    (native)
    - The minimized structure,                                             (minimized)
    - The structure as predicted by the model without relaxation, and      (af2norelax)
    - The structure as predicted by the model with relaxation.             (af2relax)

    The energies are computed using the OpenMM package.
     
    The script takes as input the path to a folder containing the files described above.
"""

from glob import glob
import os
import sys
from typing import List

import sidechainnet as scn
from sidechainnet.dataloaders.SCNProtein import SCNProtein
import openmm.unit as unit


def load_proteins_from_folder(pnid, folder, scncomplete, scnmin):
    """Load the four structures from a folder."""
    # Load the proteins
    try:
        af2norelax_fn = glob(os.path.join(folder, f"{pnid}*_unrelaxed.pdb"))[0]
        af2relax_fn = glob(os.path.join(folder, f"{pnid}*_relaxed.pdb"))[0]
    except IndexError:
        print(f"Could not find files for {pnid} in {folder}")
        return None

    af2relax = SCNProtein.from_pdb(af2relax_fn, pdbid=f"{pnid}_af2relax")
    af2norelax = SCNProtein.from_pdb(af2norelax_fn, pdbid=f"{pnid}_af2norelax")
    minimized = scnmin[pnid]
    native = scncomplete[pnid]

    # Print the lengths of each protein structure
    print(f"{pnid} lengths: native={len(native)}, minimized={len(minimized)}, "
          f"af2norelax={len(af2norelax)}, af2relax={len(af2relax)}")

    # At this point, the AF2 structures may have different lengths than the native and minimized structures.
    # This is because AF2 makes predictions for all residues, while the minimized structures
    # only contain residues that are present in the PDB. We therefore need to trim the AF2 structures
    # to match the lengths of the minimized structures.
    af2relax = trim_to_length(af2relax, minimized, native.mask)
    af2norelax = trim_to_length(af2norelax, minimized, native.mask)

    assert len(af2relax) == len(af2norelax) == len(minimized), "Lengths do not match."

    if len(minimized) != len(af2norelax) or len(minimized) != len(af2relax):
        # print(f"Length mismatch for {pnid}")
        return None

    result = {
        "native": native,
        "af2norelax": af2norelax,
        "af2relax": af2relax,
        "minimized": minimized
    }
    return result


def trim_to_length(protein, reference, mask):
    """Trim a protein to match the length of another protein."""
    if len(protein) == len(reference):
        return protein
    elif len(protein) < len(reference):
        raise ValueError("Reference protein is longer than protein to be trimmed.")
    else:
        assert len(protein) == len(mask), "Lengths do not match."
        protein.mask = mask
        protein.trim_edges()
        return protein


def main():
    # Parse arguments
    if len(sys.argv) != 5:
        print("Usage: python evaluate_energy.py <prediction_dir_path> "
              "<scn_complete_path> <scn_minimized_path> <out_csv_path>")
        sys.exit(1)
    prediction_dir = sys.argv[1]
    scn_path = sys.argv[2]
    scnmin_path = sys.argv[3]
    out_csv_path = sys.argv[4]

    # Load SidechainNet min
    scnmin = scn.load(local_scn_path=scnmin_path, trim_edges=False)

    # Load SidechainNet complete
    scncomplete = scn.load(local_scn_path=scn_path, trim_edges=False)

    # Compute energies and write to CSV for comparison
    with open(out_csv_path, "a") as f:
        # f.write(
        # "pnid,native,minimized,native_eloss,minimized_eloss,af2norelax,af2relax\n")
        skip_start = True
        failed_id = "1S1N_1_A"
        for p in scnmin:
            if skip_start and p.id != failed_id:
                continue
            elif skip_start and p.id == failed_id:
                skip_start = False
                continue
            proteins = load_proteins_from_folder(p.id, prediction_dir, scncomplete,
                                                 scnmin)
            if proteins is None:
                continue
            energies = compute_energies(proteins)
            row = f"{p.id},{energies['native']},{energies['minimized']}," + \
                f"{energies['native_eloss']},{energies['minimized_eloss']}," + \
                f"{energies['af2norelax']},{energies['af2relax']}\n"
            f.write(row)
            print(row.strip())

            f.flush()
            # print(f"{p.id} written successfully.")


def compute_energies(proteins: dict):
    """Compute the energies of the four structures."""
    energies = {}
    for name, protein in proteins.items():
        if name == "native":
            protein.trim_edges()
            assert len(protein) == len(proteins["minimized"]), "Lengths do not match."
        try:
            energies[name] = protein.get_energy(
                add_missing=True,
                add_hydrogens_via_openmm=True).value_in_unit(unit.kilojoule_per_mole)
        except ValueError:
            energies[name] = None

        if name in ["native", "minimized"]:
            protein.openmm_initialized = False
            protein.torch()
            protein.fastbuild(add_hydrogens=True, inplace=True)
            try:
                energies[name + "_eloss"] = protein.get_energy_loss()
            except ValueError:
                energies[name + "_eloss"] = None

    return energies


if __name__ == "__main__":
    sys.argv = [
        "/home/jok120/sidechainnet/sidechainnet/research/evaluate_energy/evaluate_energy.py",  # This script
        "/scr/experiments/221114/out2/predictions",  # AF2 predictions
        "/home/jok120/scn221001/sidechainnet_casp12_100.pkl",  # Complete scn path
        "/home/jok120/scnmin221013/scn_minimized.pkl",  # Minimized scn path
        "/home/jok120/sidechainnet/sidechainnet/research/evaluate_energy/result04.csv"  # Output CSV
    ]
    main()
