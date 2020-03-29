"""
    Using ProteinNet as a guide, this script creates a new dataset that adds sidechain atom information.
    It retains data splits, sequences, and masks but recomputes each structure's coordinates so that
    sidechain atoms may be recorded. It also saves the entirety of the data as a single Python dictionary.

    Author: Jonathan King
    Date:   July 21, 2019
"""

import argparse
import datetime
import multiprocessing
import os
import re
import sys

import numpy as np
import prody as pr
import torch
import tqdm

from protein_transformer.dataset import MAX_SEQ_LEN

sys.path.append(".")
from protein_transformer.protein.structure_utils import angle_list_to_sin_cos, get_seq_and_masked_coords_and_angles, \
    no_nans_infs_allzeros, parse_astral_summary_file, get_chain_from_astral_id, GLOBAL_PAD_CHAR
from protein_transformer.protein.structure_exceptions import  NonStandardAminoAcidError, SequenceError, \
    ContigMultipleMatchingError, ShortStructureError, MissingAtomsError, NoneStructureError
from proteinnet_errors import ERRORS
import parse_raw_proteinnet

pr.confProDy(verbosity='error')


def get_chain_from_trainid(proteinnet_id):
    """
    Given a ProteinNet ID of a training or validation set item, this function returns the associated
    ProDy-parsed chain object. "1A9U_2_A"
    """
    # Try parsing the ID as a PDB ID. If it fails, assume it's an ASTRAL ID.
    try:
        pdbid, model_id, chid = proteinnet_id.split("_")
        if "#" in pdbid:
            pdbid = pdbid.split("#")[1]
    except ValueError:
        try:
            pdbid, astral_id = proteinnet_id.split("_")
            return get_chain_from_astral_id(astral_id.replace("-", "_"), ASTRAL_ID_MAPPING)
        except KeyError:
            return ERRORS["MISSING_ASTRAL_IDS"]
        except ValueError:
            return ERRORS["FAILED_ASTRAL_IDS"]
        except:
            return ERRORS["FAILED_ASTRAL_IDS"]

    # Continue loading the chain, given the PDB ID
    try:
        chain = pr.parsePDB(pdbid, chain=chid)
    except:
        try:
            chain = pr.parseCIF(pdbid, chain=chid) # changed pr.parsePDB to pr.parseCIF, removed heirarchal view
        except AttributeError:
            return ERRORS["PARSING_ERROR_ATTRIBUTE"]
        except pr.proteins.pdbfile.PDBParseError:
            return ERRORS["PARSING_ERROR"]
        except OSError:
            return ERRORS["PARSING_ERROR_OSERROR"]
        except Exception as e:
            return ERRORS["UNKNOWN_EXCEPTIONS"]

    if chain is None:
        print(proteinnet_id)
        return ERRORS["NONE_CHAINS"]
    # Attempt to select a coordset
    try:
        if chain.numCoordsets() > 1:
            chain.setACSIndex(int(model_id))
    except IndexError:
        return ERRORS["COORDSET_INDEX_ERROR"]

    return chain


def get_chain_from_testid(proteinnet_id):
    """
    Given a ProteinNet ID of a test set item, this function returns the associated
    ProDy-parsed chain object.
    """
    # TODO: assert existence of test/targets at start of script
    category, caspid = proteinnet_id.split("#")
    try:
        pdb_hv = pr.parsePDB(os.path.join(args.input_dir, "targets", caspid + ".pdb")).getHierView() # TODO change to pr.parseCIF?
    except AttributeError:
        return ERRORS["TEST_PARSING_ERRORS"]
    try:
        assert pdb_hv.numChains() == 1
    except:
        print("Only a single chain should be parsed from the CASP targ PDB.")
        pass
    chain = next(iter(pdb_hv))
    return chain


def get_chain_from_proteinnetid(pdbid_chain):
    """
    Determines whether or not a PN id is a test or training id and calls the corresponding method.
    """
    # If the ProteinNet ID is from the test set
    if "TBM#" in pdbid_chain or "FM#" in pdbid_chain or "TBM-hard" in pdbid_chain or "FM-hard" in pdbid_chain:
        chain = get_chain_from_testid(pdbid_chain)
    # If the ProteinNet ID is from the train or validation set
    else:
        chain = get_chain_from_trainid(pdbid_chain)
    return chain


def get_proteinnet_seq_from_id(pnid):
    """
    Given a ProteinNet ID, this method returns the associated primary AA sequence.
    """
    if "#" not in pnid:
        true_seq = PN_TRAIN_DICT[pnid]["primary"]
    elif "TBM#" in pnid or "FM#" in pnid or "TBM-hard" in pnid:
        true_seq = PN_TEST_DICT[pnid]["primary"]
    else:
        true_seq = PN_VALID_DICT[pnid]["primary"]
    return true_seq


def work(pdbid_chain):
    """
    For a single PDB ID with chain, i.e. ('1A9U_1_A'), fetches that PDB chain from the PDB and
    computes its angles, coordinates, and sequence. The angles and coordinates contain
    GLOBAL_PAD_CHARs where there was missing data.
    """
    true_seq = get_proteinnet_seq_from_id(pdbid_chain) # TODO replace this function that returns chain from file w/ seq
    chain = get_chain_from_proteinnetid(pdbid_chain)  # Returns ProDy chain object
    if not chain:
        return ERRORS["NONE_STRUCTURE_ERRORS"]
    elif type(chain) == int:
        # This indicates there was an issue parsing the chain
        return chain
    try:
        dihedrals_coords_sequence = get_seq_and_masked_coords_and_angles(chain, true_seq)
    except NonStandardAminoAcidError:
        return ERRORS["NSAA_ERRORS"]
    except NoneStructureError:
        return ERRORS["NONE_STRUCTURE_ERRORS"]
    except ContigMultipleMatchingError:
        return ERRORS["MULTIPLE_CONTIG_ERRORS"]
    except ShortStructureError:
        return ERRORS["SHORT_ERRORS"]
    except MissingAtomsError:
        return ERRORS["MISSING_ATOMS_ERROR"]
    except SequenceError:
        print("Not fixed.", pdbid_chain)
        return ERRORS["SEQUENCE_ERRORS"]

    # If we've made it this far, we can unpack the data and return it
    dihedrals, coords, sequence = dihedrals_coords_sequence

    return dihedrals, coords, sequence, pdbid_chain


def unpack_processed_results(results, pnids):
    """
    Given an iterable of processed results containing angles, sequences, and PDB IDs,
    this function separates out the components (sequences as one-hot vectors, angle matrices,
    and PDB IDs) iff all were successfully preprocessed.
    """
    all_ohs = []
    all_angs = []
    all_crds = []
    all_ids = []
    c = 0
    for r, pnid in zip(results, pnids):
        if type(r) == int:
            # PDB failed to download
            ERRORS.count(r, pnid)
            continue
        ang, coords, seq, i = r
        if  no_nans_infs_allzeros(ang) and no_nans_infs_allzeros(coords):
            all_ohs.append(seq)
            all_angs.append(ang)
            all_crds.append(coords)
            all_ids.append(i)
            c += 1
    print(f"{(c * 100) / len(results):.1f}% of chains parsed. ({c}/{len(results)})")
    return all_ohs, all_angs, all_crds, all_ids


def validate_data_dict(data):
    """
    Performs several checks on dictionary before saving.
    """
    # Assert size of each data subset matches
    train_len = len(data["train"]["seq"])
    test_len = len(data["test"]["seq"])
    items_recorded = ["seq", "ang", "ids", "crd"]
    for num_items, subset in zip([train_len, test_len], ["train", "test"]):
        assert all([l == num_items
                    for l in map(len, [data[subset][k]
                                       for k in items_recorded])]), f"{subset} lengths don't match."

    for split in VALID_SPLITS:
        valid_len = len(data[f"valid-{split}"]["seq"])
        assert all([l == valid_len for l in map(len, [data[f"valid-{split}"][k] for k in ["ang", "ids", "crd"]])]), \
            "Valid lengths don't match."


def create_data_dict(train_seq, test_seq, train_ang, test_ang, train_crd, test_crd, train_ids, test_ids, all_validation_data):
    """
    Given split data along with the query information that generated it, this function saves the
    data as a Python dictionary, which is then saved to disk using torch.save.
    See commit  d1935a0869720f85c00824f3aecbbfc6b947711c for a method that saves all relevant information.
    """
    # Sort data
    train_ang, train_seq, train_crd, train_ids = sort_data(train_ang, train_seq, train_crd, train_ids)
    test_ang, test_seq, test_crd, test_ids = sort_data(test_ang, test_seq, test_crd, test_ids)

    # Create a dictionary data structure, using the sin/cos transformed angles
    data = {"train": {"seq": train_seq,
                      "ang": angle_list_to_sin_cos(train_ang),
                      "ids": train_ids,
                      "crd": train_crd},
            "test": {"seq": test_seq,
                     "ang": angle_list_to_sin_cos(test_ang),
                     "ids": test_ids,
                     "crd": test_crd},
            "settings": {"max_len": max(map(len, train_seq + test_seq)),
                         "pad_char": GLOBAL_PAD_CHAR},
            "description": {f"ProteinNet {CASP_VERSION.upper()}"},
            # To parse date later, use datetime.datetime.strptime(date, "%I:%M%p on %B %d, %Y")
            "date": datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")}
    max_val_len = 0
    for split, (seq_val, ang_val, crd_val, ids_val) in all_validation_data.items():
        ang_val, seq_val, crd_val, ids_val = sort_data(ang_val, seq_val, crd_val, ids_val)
        data[f"valid-{split}"] = {}
        data[f"valid-{split}"]["seq"] = seq_val
        data[f"valid-{split}"]["ang"] = angle_list_to_sin_cos(ang_val)
        data[f"valid-{split}"]["crd"] = crd_val
        data[f"valid-{split}"]["ids"] = ids_val
        max_split_len = max(data["settings"]["max_len"], max(map(len, seq_val)))
        max_val_len = max_split_len if max_split_len > max_val_len else max_val_len
    data["settings"]["max_len"] = max(list(map(len, train_seq + test_seq)) + [max_val_len])

    data["settings"]["bin-data"] = bin_sequence_data(train_seq, maxlen=MAX_SEQ_LEN)
    data["settings"]["angle_means"] = compute_angle_means(data)
    validate_data_dict(data)
    return data


def compute_angle_means(data):
    """ Computes and retuns the mean of the training data angle matrices. """
    train_angles_sincos = np.concatenate(data["train"]["ang"])
    means = np.nanmean(train_angles_sincos, axis=0)
    return means


def bin_sequence_data(seqs, maxlen):
    """
    Given a list of sequences and a maximum training length, this function
    bins the sequences by their lengths (using numpy's 'auto' parameter),
    and then records the histogram information, as well as some statistics.
    This information is returned as a dictionary.

    This function allows the user to avoid computing this information at the
    start of each training run.
    """
    lens = list(map(lambda x: len(x) if len(x) <= maxlen else maxlen, seqs))
    hist_counts, hist_bins = np.histogram(lens, bins="auto")
    hist_bins = hist_bins[1:]  # make each bin define the rightmost value in each bin, ie '( , ]'.
    bin_probs = hist_counts / hist_counts.sum()
    bin_map = {}

    # Compute a mapping from bin number to index in dataset
    seq_i = 0
    bin_j = 0
    while seq_i < len(seqs):
        if lens[seq_i] <= hist_bins[bin_j]:
            try:
                bin_map[bin_j].append(seq_i)
            except KeyError:
                bin_map[bin_j] = [seq_i]
            seq_i += 1
        else:
            bin_j += 1

    return {"hist_counts" : hist_counts,
            "hist_bins": hist_bins,
            "bin_probs" : bin_probs,
            "bin_map": bin_map,
            "bin_max_len": maxlen}


def add_proteinnetID_to_idx_mapping(data):
    """
    Given an already processes ProteinNet data dictionary, this function adds
    a mapping from ProteinNet ID to the subset and index number where that
    protein can be looked up in the current dictionary. Useful if you'd like
    to quickly extract a certain protein.
    """
    d = {}
    for subset in ["train", "test"] + [f"valid-{split}" for split in VALID_SPLITS]:
        for idx, pnid in enumerate(data[subset]["ids"]):
            d[pnid] = {"subset": subset, "idx": idx}

    data["pnids"] = d
    return data


def sort_data(angs, seqs, crds, ids):
    """
    Sorts inputs by length, with shortest first.
    """
    sorted_len_indices = [a[0] for a in sorted(enumerate(angs),
                                               key=lambda x:x[1].shape[0],
                                               reverse=False)]
    seqs = [seqs[i] for i in sorted_len_indices]
    crds = [crds[i] for i in sorted_len_indices]
    angs = [angs[i] for i in sorted_len_indices]
    ids = [ids[i] for i in sorted_len_indices]

    return angs, seqs, crds, ids


def group_validation_set(vset_ids):
    """
    Given a list of validation set ids, (i.e. 70#1A9U_1_A), this returns a dictionary that maps each split
    to the list of PDBS in that split.
    >>> vids = ["70#1A9U_1_A", "30#1Z3F_1_B"]
    >>> group_validation_set(vids)
    {70: "70#1A9U_1_A", 30:"30#1Z3F_1_B"}
    """
    # Because there are several validation sets, we group IDs by their seq identity for use later
    valid_ids_grouped = {k: [] for k in VALID_SPLITS}
    for vid in vset_ids:
        group = int(vid[:2])
        valid_ids_grouped[group].append(vid)
    return valid_ids_grouped


def save_data_dict(data):
    """
    Saves a Python dictionary containing all training data to disk via Pickle or PyTorch.
    """
    if not args.out_file:
        args.out_file = "../data/proteinnet/" + CASP_VERSION + "_" + SUFFIX + ".pt"
    torch.save(data, args.out_file)
    print(f"Data saved to {args.out_file}.")


def main():
    lim = args.limit
    global PN_TRAIN_DICT, PN_VALID_DICT, PN_TEST_DICT
    train_pdb_ids, valid_ids, test_casp_ids = parse_raw_proteinnet.parse_raw_proteinnet(args.input_dir, TRAIN_FILE)
    print("IDs fetched.")
    PN_TRAIN_DICT, PN_VALID_DICT, PN_TEST_DICT = torch.load(
        os.path.join(args.input_dir, "torch", TRAIN_FILE)), torch.load(
        os.path.join(args.input_dir, "torch", "validation.pt")), torch.load(
        os.path.join(args.input_dir, "torch", "testing.pt"))
    print(len(train_pdb_ids), len(valid_ids), len(test_casp_ids))
    # Download and preprocess all data from PDB IDs
    with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
        train_results = list(tqdm.tqdm(p.imap(work, train_pdb_ids[:lim]), total=len(train_pdb_ids[:lim]), dynamic_ncols=True))
    if lim:
        vlim = 1
    else:
        vlim = None
    valid_result_meta = {}
    for split, vids in group_validation_set(valid_ids).items():
        valid_result_meta[split] = []
        for vid in tqdm.tqdm(vids[:vlim], dynamic_ncols=True):
            valid_result_meta[split].append(work(vid))

    test_results = []
    for tid in tqdm.tqdm(test_casp_ids[:vlim], dynamic_ncols=True):
        test_results.append(work(tid))

    print("Structures processed.")
    

    # Unpack results
    print("Training set:\t", end="")
    train_ohs, train_angs, train_strs, train_ids = unpack_processed_results(train_results, train_pdb_ids[:lim])
    for (split, results), vids in zip(valid_result_meta.items(), group_validation_set(valid_ids).values()):
        print(f"Valid set {split}%:\t", end="")
        valid_result_meta[split] = unpack_processed_results(results, vids[:vlim])
    print("Test set:\t\t", end="")
    test_ohs, test_angs, test_strs, test_ids = unpack_processed_results(test_results, test_casp_ids[:vlim])
    ERRORS.summarize()

    # Split into train, test and validation sets. Report sizes.
    data = create_data_dict(train_ohs, test_ohs, train_angs, test_angs, train_strs, test_strs, train_ids, test_ids,
                            valid_result_meta)
    data = add_proteinnetID_to_idx_mapping(data)
    save_data_dict(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Converts a ProteinNet directory of raw records into a dataset for all"
                    "atom protein structure prediction.")
    parser.add_argument('input_dir', type=str, help='Path to ProteinNet raw records directory.')
    parser.add_argument('-o', '--out_file', type=str, help='Path to output file (.tch file)')
    parser.add_argument('-l', '--limit', type=int, default=None, help='Limit size of training set for debugging.')
    parser.add_argument("--pdb_dir", default=os.path.expanduser("~/pdb/"), type=str,
                        help="Path for ProDy-downloaded PDB files.")
    parser.add_argument('--training_set', type=int, default=100, help='Which thinning of the training set to parse. '
                                                                      '{30,50,70,90,95,100}. Default 100.')
    args = parser.parse_args()

    VALID_SPLITS = [10, 20, 30, 40, 50, 70, 90]
    TRAIN_FILE = f"training_{args.training_set}.pt"
    PN_TRAIN_DICT, PN_VALID_DICT, PN_TEST_DICT = None, None, None
    ASTRAL_FILE = "../data/astral_pdb_map.txt" # combined previous versions of dir.des.scope.2.xx-stable.txt into one big dict
    ASTRAL_ID_MAPPING = parse_astral_summary_file(ASTRAL_FILE)
    SUFFIX = str(datetime.datetime.today().strftime("%y%m%d")) + f"_{args.training_set}"
    match = re.search(r"casp\d+", args.input_dir, re.IGNORECASE)
    assert match, "The input_dir is not titled with 'caspX'."
    CASP_VERSION = match.group(0)

    pr.pathPDBFolder(args.pdb_dir)  # Set PDB download location
    np.set_printoptions(suppress=True)  # suppresses scientific notation when printing
    np.set_printoptions(threshold=sys.maxsize)  # suppresses '...' when printing

    try:
        main()
    except Exception as e:
        ERRORS.summarize()
        raise e
