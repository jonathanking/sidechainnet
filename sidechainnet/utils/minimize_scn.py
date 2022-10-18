import copy
import datetime
from glob import glob
from multiprocessing import Pool
import os
import pickle
import sys
import traceback
from tqdm import tqdm

import sidechainnet as scn
from sidechainnet.dataloaders.SCNDataset import SCNDataset
from sidechainnet.dataloaders.SCNProtein import SCNProtein
from sidechainnet.utils.minimizer import SCNMinimizer
from sidechainnet.utils.organize import EMPTY_SPLIT_DICT

UNMIN_PATH = ""


def setup(datapath, unmin_path):
    """Create a directory containing individual SCN Proteins as pickle files."""
    global UNMIN_PATH
    UNMIN_PATH = unmin_path
    os.makedirs(os.path.join(unmin_path), exist_ok=True)
    d = scn.load(local_scn_path=datapath,
                 scn_dataset=True,
                 filter_by_resolution=False,
                 complete_structures_only=False)
    # d.filter(lambda p: len(p) < 15)
    with Pool() as p:
        _ = list(tqdm(p.imap(do_pickle, d), total=len(d)))


def do_pickle(protein):
    """Pickle a protein object and save to UNMIN_PATH/{pid}.pkl."""
    protein.pickle(os.path.join(UNMIN_PATH, f"{protein.id}.pkl"))


def process_index(index, unmin_path, min_path):
    """Minimize a single protein by its index in the sorted list of files."""
    parent, _ = os.path.split(min_path)
    os.makedirs(os.path.join(parent, "failed"), exist_ok=True)
    os.makedirs(min_path, exist_ok=True)

    filename = get_filename_from_index_path(index, unmin_path)
    output_path = os.path.join(min_path, os.path.basename(filename))

    # Load unminimized protein
    with open(filename, "rb") as f:
        datadict = pickle.load(f)
    protein = SCNProtein(**datadict)

    try:
        print(f"Minimizing Protein {protein.id}.")
        process_protein_obj(protein, output_path)
    except ValueError as e:
        print(e, end="\n\n")
        traceback.print_exc()
        with open(os.path.join(parent, "failed", protein.id + ".txt"), "w") as f:
            f.write(traceback.format_exc())
        exit(0)
    protein.numpy()
    protein.pickle(output_path)
    protein.to_pdb(output_path.replace(".pkl", ".pdb"))
    print("Minimized protein written to", output_path)


def process_protein_obj(protein, output_path=None):
    """Minimize a single protein object with LBFGS."""
    m = SCNMinimizer()
    m.minimize_scnprotein(protein, use_sgd=False, verbose=True, path=output_path)


def get_filename_from_index_path(index, path):
    filenames = sorted(glob(os.path.join(path, "*.pkl")))
    try:
        return filenames[index]
    except IndexError:
        print(f"Index {index} unavailable out of {len(filenames)} matching files.")
        exit(1)


def cleanup(min_path):
    """Create a new SidechainNet data file by combining all individual files in path."""
    filenames = sorted(glob(os.path.join(min_path, "*.pkl")))
    parent, _ = os.path.split(min_path)

    data = {"date": datetime.datetime.now().strftime("%I:%M%p %b %d, %Y"), "settings": {}}

    for fn in tqdm(filenames):
        with open(fn, "rb") as f:
            datadict = pickle.load(f)
        p = SCNProtein(**datadict)
        if p.split not in data:
            data[p.split] = copy.deepcopy(EMPTY_SPLIT_DICT)
        p.numpy()
        data[p.split]["ang"].append(p.angles)
        data[p.split]["seq"].append(p.seq)
        data[p.split]["ids"].append(p.id)
        data[p.split]["evo"].append(p.evolutionary)
        data[p.split]["msk"].append(p.mask)
        data[p.split]["crd"].append(p.coords)
        data[p.split]["sec"].append(p.secondary_structure)
        data[p.split]["res"].append(p.resolution)
        data[p.split]["ums"].append(p.unmodified_seq)
        data[p.split]["mod"].append(p.is_modified)

    dataset = SCNDataset(data)
    outpath = os.path.join(parent, "scn_minimized.pkl")
    dataset.pickle(outpath)
    print(f"Cleanup complete. Please see {outpath}.")


if __name__ == "__main__":
    _, step, datapath, unmin_path, min_path, index = sys.argv
    if step == "setup":
        setup(datapath, unmin_path)
    elif step == "process_index":
        process_index(int(index), unmin_path, min_path)
    elif step == "cleanup":
        cleanup(min_path)
