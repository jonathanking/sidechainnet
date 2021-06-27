"""Contains methods for organizing SidechainNet data into a Python dictionary."""

import copy
import datetime
import os
import pickle
import re

import numpy as np

from sidechainnet.utils.download import determine_pnid_type

EMPTY_SPLIT_DICT = {
    "seq": [],
    "ang": [],
    "ids": [],
    "evo": [],
    "msk": [],
    "crd": [],
    "sec": [],
    "res": [],
    "ums": [],
    "mod": []
}


def validate_data_dict(data):
    """Performs several sanity checks on the data dict before saving."""
    from sidechainnet.utils.download import VALID_SPLITS
    # Assert size of each data subset matches
    train_len = len(data["train"]["seq"])
    test_len = len(data["test"]["seq"])
    items_recorded = ["seq", "ang", "ids", "crd", "msk", "evo"]
    for num_items, subset in zip([train_len, test_len], ["train", "test"]):
        assert all([
            l == num_items for l in map(len, [data[subset][k] for k in items_recorded])
        ]), f"{subset} lengths don't match."

    for vsplit in VALID_SPLITS:
        valid_len = len(data[vsplit]["seq"])
        assert all([
            l == valid_len
            for l in map(len, [data[vsplit][k] for k in ["ang", "ids", "crd"]])
        ]), "Valid lengths don't match."


def create_empty_dictionary():
    """Create an empty SidechainNet dictionary ready to hold SidechainNet data."""
    from sidechainnet.utils.download import VALID_SPLITS

    data = {
        "train": copy.deepcopy(EMPTY_SPLIT_DICT),
        "test": copy.deepcopy(EMPTY_SPLIT_DICT),
        # To parse date, use datetime.datetime.strptime(date, "%I:%M%p on %B %d, %Y")
        "date": datetime.datetime.now().strftime("%I:%M%p %b %d, %Y"),
        "settings": dict()
    }

    validation_subdict = {
        vsplit: copy.deepcopy(EMPTY_SPLIT_DICT) for vsplit in VALID_SPLITS
    }
    data.update(validation_subdict)

    return data


def get_proteinnetIDs_by_split(casp_version, thinning, custom_ids=None):
    """Returns a dict of ProteinNet IDs organized by data split (train/test/valid)."""
    from sidechainnet.create import get_proteinnet_ids
    if custom_ids is not None:
        ids_datasplit = [(_id, determine_pnid_type(_id)) for _id in custom_ids]
        ids = {"train": [], "valid": [], "test": []}
        for _id, split in ids_datasplit:
            ids[split].append(_id)
        return ids

    if thinning == "debug":
        thinning = 100

    ids = {
        "train": get_proteinnet_ids(casp_version, "train", thinning=thinning),
        "valid": get_proteinnet_ids(casp_version, "valid"),
        "test": get_proteinnet_ids(casp_version, "test")
    }

    return ids


def organize_data(scnet_data,
                  casp_version,
                  thinning,
                  is_debug=False,
                  description=None,
                  custom_ids=None):
    """Given an unsorted Sidechainnet data dict, organizes into ProteinNet data splits.

    Args:
        scnet_data: A dictionary mapping ProteinNet ids (pnids) to data recorded by
            SidechainNet ('seq', 'ang', 'crd', 'evo', 'msk').
        casp_version: A string describing the CASP version of this dataset.
        thinning: An integer representing the training set thinning.
        is_debug: A bool. If True, sample 200 training set IDs.
        description: A string describing the dataset.
        custom_ids: (optional) A list of custom ProteinNet IDs to use for this dataset.

    Returns:
        A Python dictionary containing SidechainNet data, but this time, organized
        and divided into the data splits specified by ProteinNet.
    """
    from sidechainnet.utils.download import DATA_SPLITS
    # First, we need to determine which pnids belong to which data split.
    ids = get_proteinnetIDs_by_split(casp_version, thinning, custom_ids)

    # Next, we create the empty dictionary for storing the data, organized by data splits
    organized_data = create_empty_dictionary()

    # Now, we organize the data by its data splits
    n_proteins = 0
    for split in ["train", "test", "valid"]:
        if split == "train" and is_debug:
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
            organized_data[realsplit]['ums'].append(scnet_data[pnid]['ums'])
            organized_data[realsplit]['mod'].append(scnet_data[pnid]['mod'])
            organized_data[realsplit]['ids'].append(pnid)
            n_proteins += 1

    # Sort each split of data by length, ascending
    for split in DATA_SPLITS:
        organized_data[split] = sort_datasplit(organized_data[split])

    # Add settings
    organized_data["description"] = description
    organized_data["settings"]["casp_version"] = int(
        casp_version) if (isinstance(casp_version, int) or casp_version.isnumeric()) else casp_version
    organized_data["settings"]["thinning"] = int(
        thinning) if (isinstance(thinning, int) or thinning.isnumeric()) else thinning
    organized_data["settings"]["n_proteins"] = n_proteins
    organized_data["settings"]["angle_means"] = compute_angle_means(
        organized_data['train']['ang'])
    organized_data["settings"]["lengths"] = np.sort(
        np.asarray(list(map(len, (v['seq'] for k, v in scnet_data.items())))))
    organized_data['settings']['max_length'] = organized_data["settings"]["lengths"].max()

    print(f"{n_proteins} included in CASP {casp_version} ({thinning}% thinning).")

    validate_data_dict(organized_data)

    return organized_data


def get_validation_split_identifiers_from_pnid_list(pnids):
    """Return a sorted list of validation set identifiers given a list of ProteinNet IDs.

    Args:
        pnids (list): List of ProteinNet-formated IDs (90#1A9U_1_A)

    Returns:
        List: List of validation set identifiers present in the list of pnids.

    Example:
        >>> pnids = ['40#1XHN_1_A', '10#2MEM_1_A', '90#3EOI_1_A']
        >>> get_validation_split_identifiers_from_pnid_list(pnids)
        [10, 40, 90]
    """
    matches = (re.match(r"(\d+)#\S+", s) for s in pnids)
    matches = set((m.group(1) for m in filter(lambda s: s is not None, matches)))
    return sorted(map(int, matches))


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
