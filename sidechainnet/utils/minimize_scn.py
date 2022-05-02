import copy
import datetime
from glob import glob
import os
import pickle
import sys

import sidechainnet as scn
from sidechainnet.dataloaders.SCNDataset import SCNDataset
from sidechainnet.dataloaders.SCNProtein import SCNProtein
from sidechainnet.utils.minimizer import SCNMinimizer
from sidechainnet.utils.organize import EMPTY_SPLIT_DICT


def setup(datapath, picklepath):
    """Create a directory containing individual SCN Proteins as pickle files."""
    d = scn.load(local_scn_path=datapath,
                 scn_dataset=True,
                 filter_by_resolution=False,
                 complete_structures_only=True)
    d.filter(lambda p: len(p) < 15)
    for p in d:
        p.pickle(os.path.join(picklepath, f"{p.id}.pkl"))


def cleanup(picklepath):
    """Create a new SidechainNet data file by combining all individual files in path."""
    filenames = sorted(glob(os.path.join(picklepath, "out", "*.pkl")))

    data = {"date": datetime.datetime.now().strftime("%I:%M%p %b %d, %Y"), "settings": {}}

    for fn in filenames:
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
    dataset.pickle(os.path.join(picklepath, "out", "scn_minimized.pkl"))


def process_index(index, picklepath):
    """Minimize a single protein by its index in the sorted list of files."""
    filename = get_filename_from_index_path(index, picklepath)
    with open(filename, "rb") as f:
        datadict = pickle.load(f)
    protein = SCNProtein(**datadict)
    m = SCNMinimizer()
    m.minimize_scnprotein(protein, use_sgd=False, verbose=True)
    output_path = os.path.join(picklepath, "out", os.path.basename(filename))
    os.makedirs(os.path.join(picklepath, "out"), exist_ok=True)
    protein.pickle(output_path)


def get_filename_from_index_path(index, path):
    filenames = sorted(glob(os.path.join(path, "*.pkl")))
    try:
        return filenames[index]
    except IndexError:
        print(f"Index {index} unavailable out of {len(filenames)} matching files.")
        exit(1)


if __name__ == "__main__":
    _, step, datapath, picklepath, index = sys.argv
    if step == "setup":
        setup(datapath, picklepath)
    elif step == "cleanup":
        cleanup(picklepath)
    elif step == "process_index":
        process_index(int(index), picklepath)
