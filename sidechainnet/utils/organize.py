"""Contains methods for organizing SidechainNet data into a Python dictionary."""

import copy
import datetime
import os
import pickle

import numpy as np

from sidechainnet.utils.download import VALID_SPLITS


def validate_data_dict(data):
    """Performs several sanity checks on the data dict before saving."""
    # Assert size of each data subset matches
    train_len = len(data["train"]["seq"])
    test_len = len(data["test"]["seq"])
    items_recorded = ["seq", "ang", "ids", "crd", "msk", "evo"]
    for num_items, subset in zip([train_len, test_len], ["train", "test"]):
        assert all([
            l == num_items for l in map(len, [data[subset][k] for k in items_recorded])
        ]), f"{subset} lengths don't match."

    for split in VALID_SPLITS:
        valid_len = len(data[f"valid-{split}"]["seq"])
        assert all([
            l == valid_len
            for l in map(len, [data[f"valid-{split}"][k] for k in ["ang", "ids", "crd"]])
        ]), "Valid lengths don't match."


def create_empty_dictionary():
    """Create an empty SidechainNet dictionary ready to hold SidechainNet data."""
    basic_data_entries = {
        "seq": [],
        "ang": [],
        "ids": [],
        "evo": [],
        "msk": [],
        "crd": [],
        "sec": [],
        "res": []
    }

    data = {
        "train": copy.deepcopy(basic_data_entries),
        "test": copy.deepcopy(basic_data_entries),
        # To parse date, use datetime.datetime.strptime(date, "%I:%M%p on %B %d, %Y")
        "date": datetime.datetime.now().strftime("%I:%M%p %b %d, %Y"),
        "settings": dict()
    }

    validation_subdict = {
        f"valid-{split}": copy.deepcopy(basic_data_entries) for split in VALID_SPLITS
    }
    data.update(validation_subdict)

    return data


def get_proteinnetIDs_by_split(proteinnet_dir, thinning):
    """Returns a dict of ProteinNet IDs organized by data split (train/test/valid)."""
    if thinning == "debug":
        thinning = 100
    pn_files = [
        os.path.join(proteinnet_dir, f"training_{thinning}_ids.txt"),
        os.path.join(proteinnet_dir, "validation_ids.txt"),
        os.path.join(proteinnet_dir, "testing_ids.txt")
    ]

    def parse_ids(filepath):
        with open(filepath, "r") as f:
            _ids = f.read().splitlines()
        return _ids

    ids = {
        "train": parse_ids(pn_files[0]),
        "valid": parse_ids(pn_files[1]),
        "test": parse_ids(pn_files[2])
    }

    return ids


def organize_data(scnet_data, proteinnet_dir, casp_version, thinning):
    """Given an unsorted Sidechainnet data dict, organizes into ProteinNet data splits.

    Args:
        scnet_data: A dictionary mapping ProteinNet ids (pnids) to data recorded by
            SidechainNet ('seq', 'ang', 'crd', 'evo', 'msk').
        proteinnet_dir: A string representing the path to where the preprocessed
            ProteinNet files are stored.
        casp_version: A string describing the CASP version of this dataset.

    Returns:
        A Python dictionary containing SidechainNet data, but this time, organized
        and divided into the data splits specified by ProteinNet.
    """
    # First, we need to determine which pnids belong to which data split.
    ids = get_proteinnetIDs_by_split(proteinnet_dir, thinning)

    # Next, we create the empty dictionary for storing the data, organized by data splits
    organized_data = create_empty_dictionary()

    # Now, we organize the data by its data splits
    n_proteins = 0
    for split in ["train", "test", "valid"]:
        if split == "train" and thinning == "debug":
            thinning = 0
            np.random.seed(0)
            split_ids = np.random.choice(ids[split], 200, replace=False)
        else:
            split_ids = ids[split]
        for pnid in split_ids:
            if pnid not in scnet_data:
                continue
            if 'primary' in scnet_data[pnid]:
                print(f"{pnid} had 'primary'  key.")
                del scnet_data[pnid]
                continue
            realsplit = f"valid-{pnid.split('#')[0]}" if split == "valid" else split
            organized_data[realsplit]['seq'].append(scnet_data[pnid]['seq'])
            organized_data[realsplit]['ang'].append(scnet_data[pnid]['ang'])
            organized_data[realsplit]['crd'].append(scnet_data[pnid]['crd'])
            organized_data[realsplit]['msk'].append(scnet_data[pnid]['msk'])
            organized_data[realsplit]['evo'].append(scnet_data[pnid]['evo'])
            organized_data[realsplit]['sec'].append(scnet_data[pnid]['sec'])
            organized_data[realsplit]['res'].append(scnet_data[pnid]['res'])
            organized_data[realsplit]['ids'].append(pnid)
            n_proteins += 1

    # Sort each split of data by length, ascending
    for split in ["train", "test"] + [f"valid-{vs}" for vs in VALID_SPLITS]:
        organized_data[split] = sort_datasplit(organized_data[split])

    # Add settings
    organized_data["description"] = f"SidechainNet {casp_version}"
    organized_data["settings"]["casp_version"] = int(casp_version) if casp_version != "debug" else casp_version
    organized_data["settings"]["thinning"] = int(thinning)
    organized_data["settings"]["n_proteins"] = n_proteins
    organized_data["settings"]["angle_means"] = compute_angle_means(
        organized_data['train']['ang'])
    organized_data["settings"]["lengths"] = np.sort(
        np.asarray(list(map(len, (v['seq'] for k, v in scnet_data.items())))))
    organized_data['settings']['max_length'] = organized_data["settings"]["lengths"].max()

    print(f"{n_proteins} included in CASP {casp_version} ({thinning}% thinning).")

    validate_data_dict(organized_data)

    return organized_data


def compute_angle_means(angle_list):
    """Computes mean of angle matrices in a Python list ignoring all-zero rows."""
    angles = np.concatenate(angle_list)
    angles = angles[~(angles == 0).all(axis=1)]
    return angles.mean(axis=0)


def save_data(data, path):
    """Saves an organized SidechainNet data dict to a given, local filepath."""
    with open(path, "wb") as f:
        return pickle.dump(data, f)


def load_data(path):
    """Loads SidechainNet data dict from a given, local filepath."""
    with open(path, "rb") as f:
        return pickle.load(f)


def sort_datasplit(split):
    """Sorts a single split of the SidechainNet data dict by ascending length."""
    sorted_len_indices = [
        a[0]
        for a in sorted(enumerate(split['seq']), key=lambda x: len(x[1]), reverse=False)
    ]

    for datatype in split.keys():
        split[datatype] = [split[datatype][i] for i in sorted_len_indices]

    return split
