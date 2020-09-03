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
import pickle
import re
import sys
from glob import glob

import numpy as np
import prody as pr
import tqdm

import sidechainnet
from sidechainnet.utils.astral_data import ASTRAL_SUMMARY
from sidechainnet.utils.errors import ERRORS
from sidechainnet.utils.errors import *
from sidechainnet.utils.sequence import bin_sequence_data
from sidechainnet.utils.measure import angle_list_to_sin_cos, get_seq_coords_and_angles, no_nans_infs_allzeros, GLOBAL_PAD_CHAR
from sidechainnet.utils.parse import parse_astral_summary_file, get_chain_from_astral_id

MAX_SEQ_LEN = 10_000
VALID_SPLITS = [10, 20, 30, 40, 50, 70, 90]
ASTRAL_ID_MAPPING = parse_astral_summary_file(ASTRAL_SUMMARY.splitlines())
del ASTRAL_SUMMARY


def download_sidechain_data(pnids, sidechainnet_out_dir, casp_version,
                            training_set, limit, proteinnet_in):
    """
    Downloads the sidechain data for the corresponding ProteinNet IDs.

    Args:
        training_set: Which thinning of ProteinNet to extract (30, 50, 90, etc.)
        casp_version: A string that describes the CASP version i.e. 'casp7'
        pnids: List of ProteinNet IDs to download sidechain data for
        sidechainnet_out_dir: Path to directory for saving sidechain data

    Returns:
        sc_data: Python dictionary `{pnid: {...}, ...}`
    """
    global PROTEINNET_IN_DIR
    PROTEINNET_IN_DIR = proteinnet_in
    output_name = f"sidechain-only_{casp_version}_{training_set}.pkl"
    output_path = os.path.join(sidechainnet_out_dir, output_name)
    if not os.path.exists(sidechainnet_out_dir):
        os.mkdir(sidechainnet_out_dir)

    if os.path.exists(output_path):
        print(f"Sidechain information already preprocessed ({output_path}).")
        return load_data(output_path), output_path

    if os.path.exists("errors"):
        for file in glob('errors/*.txt'):
            os.remove(file)

    sc_data, pnids_errors = get_sidechain_data(pnids, limit)

    save_data(sc_data, output_path)
    report_errors(pnids_errors, total_pnids=len(pnids[:limit]))

    # Clean up working directory
    for file in glob('*.cif'):
        os.remove(file)

    return sc_data, output_path


def report_errors(pnids_errorcodes, total_pnids):
    """ Provides a summary of errors after parsing SidechainNet data.

    Args:
        pnids_errorcodes: A list of tuples (pnid, error_code) of ProteinNet IDs
            that could not be parsed completely.
        total_pnids: Total number of pnids processed.

    Returns:
        None. Prints summary to stdout and generates files containing failed
        IDs in .errors/{ERROR_CODE}.txt

    """
    print(f"\n{total_pnids} ProteinNet IDs were processed to extract sidechain "
          f"data.")
    error_summarizer = sidechainnet.utils.errors.ProteinErrors()
    for pnid, error_code in pnids_errorcodes:
        error_summarizer.count(error_code, pnid)
    error_summarizer.summarize(total_pnids)
    if os.path.exists("errors/MODIFIED_MODEL_WARNING.txt"):
        with open("errors/MODIFIED_MODEL_WARNING.txt", "r") as f:
            model_number_errors = len(f.readlines())
            print(f"Be aware that {model_number_errors} files may be using a "
                  f"different model number than the one specified by "
                  f"ProteinNet. See errors/MODIFIED_MODEL_WARNING.txt for "
                  f"a list of these proteins.")


def get_sidechain_data(pnids, limit):
    """Acquires sidechain data for specified ProteinNet IDs.
    Args:
        pnids: List of ProteinNet IDs to download data for.
        limit: Number of IDs to process (use small value for debugging).

    Returns:
        Dictionary mapping pnids to sidechain data. For example:

        {"1a9u_A_2": {"seq": "MRYK...",
                      "ang": np.ndarray(...),
                        ...
                      }}
        Also returns a list of tuples of (pnid, error_code) for those pnids
        that failed to download.
    """
    with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
        results = list(tqdm.tqdm(p.imap(process_id, pnids[:limit]),
                                 total=len(pnids[:limit]), dynamic_ncols=True))
    all_errors = []
    all_data = dict()
    with open("errors/MODIFIED_MODEL_WARNING.txt", "a") as model_warning_file:
        for pnid, r in results:
            if type(r) == int:
                all_errors.append((pnid, r))
                continue
            if "msg" in r and r["msg"] == "MODIFIED_MODEL":
                model_warning_file.write(f"{pnid}\n")
                del r["msg"]
            all_data[pnid] = r
    return all_data, all_errors


def process_id(pnid):
    """ Creates dictionary of sidechain data for a single ProteinNet ID.

    For a single ProteinNet ID i.e. ('1A9U_1_A'), fetches that PDB chain
    from the PDB and extracts its angles, coordinates, and sequence. Missing
    data is padded with GLOBAL_PAD_CHARs.
    Args:
        pnid: A ProteinNet ID, i.e. '1A9U_1_A'.

    Returns:
        Dictionary of relevant data for that ID.

    """
    message = None
    pnid_type = determine_pnid_type(pnid)
    chain = get_chain_from_proteinnetid(pnid, pnid_type)
    if not chain:
        return pnid, ERRORS["NONE_STRUCTURE_ERRORS"]
    elif type(chain) == tuple and type(chain[1]) == int:
        # This indicates there was an issue parsing the chain
        return pnid, chain[1]
    elif type(chain) == tuple and chain[1] == "MODIFIED_MODEL":
        message = "MODIFIED_MODEL"
        chain = chain[0]
    try:
        dihedrals_coords_sequence = get_seq_coords_and_angles(chain)
    except NonStandardAminoAcidError:
        return pnid, ERRORS["NSAA_ERRORS"]
    except NoneStructureError:
        return pnid, ERRORS["NONE_STRUCTURE_ERRORS"]
    except ContigMultipleMatchingError:
        return pnid, ERRORS["MULTIPLE_CONTIG_ERRORS"]
    except ShortStructureError:
        return pnid, ERRORS["SHORT_ERRORS"]
    except MissingAtomsError:
        return pnid, ERRORS["MISSING_ATOMS_ERROR"]
    except SequenceError:
        print("Not fixed.", pnid)
        return pnid, ERRORS["SEQUENCE_ERRORS"]

    # If we've made it this far, we can unpack the data and return it
    dihedrals, coords, sequence = dihedrals_coords_sequence
    data = {"ang": dihedrals, "crd": coords, "seq": sequence}
    if message:
        data["msg"] = message
    return pnid, data


def determine_pnid_type(pnid):
    """Returns the 'type' of a ProteinNet ID (i.e. train, valid, test, ASTRAL).

    Args:
        pnid: ProteinNet ID string.

    Returns:
        The 'type' of ProteinNet ID as a string.
    """
    if "TBM#" in pnid or "FM#" in pnid or "TBM-hard" in pnid or "FM-hard" in pnid:
        return "test"

    if pnid.count("_") == 1:
        is_astral = "_astral"
    else:
        is_astral = ""

    if "#" in pnid:
        return "valid" + is_astral
    else:
        return "train" + is_astral


def get_chain_from_trainid(pnid):
    """ Return a ProDy chain object for a ProteinNet ID. Assumes train/valid ID.

    Args:
        pnid: ProteinNet ID

    Returns:
        ProDy chain object corresponding to ProteinNet ID.
    """
    modified_model_number = False
    # Try parsing the ID as a PDB ID. If it fails, assume it's an ASTRAL ID.
    try:
        pdbid, chnum, chid = pnid.split("_")
        chnum = int(chnum)
        # If this is a validation set pnid, separate the annotation from the ID
        if "#" in pdbid:
            pdbid = pdbid.split("#")[1]
    except ValueError:
        try:
            pdbid, astral_id = pnid.split("_")
            return get_chain_from_astral_id(astral_id.replace("-", "_"),
                                            ASTRAL_ID_MAPPING)
        except KeyError:
            return pnid, ERRORS["MISSING_ASTRAL_IDS"]
        except ValueError:
            return pnid, ERRORS["FAILED_ASTRAL_IDS"]
        except:
            return pnid, ERRORS["FAILED_ASTRAL_IDS"]

    # Continue loading the chain, given the PDB ID
    use_pdb = True
    try:
        chain = pr.parsePDB(pdbid, chain=chid, model=chnum)
        if not chain:
            chain = pr.parseCIF(pdbid, chain=chid, model=chnum)
            use_pdb = False
    # If the file is too large, then we can download the CIF instead
    except OSError:
        try:
            chain = pr.parseCIF(pdbid, chain=chid, model=chnum)
            use_pdb = False
        except IndexError:
            try:
                chain = pr.parseCIF(pdbid, chain=chid, model=1)
                use_pdb = False
                modified_model_number = True
            except Exception as e:
                print(e)
                return pnid, ERRORS["PARSING_ERROR_OSERROR"]
        except Exception as e: # EOFERROR
            print(e)
            return pnid, ERRORS["PARSING_ERROR_OSERROR"]
    except AttributeError:
        return pnid, ERRORS["PARSING_ERROR_ATTRIBUTE"]
    except (pr.proteins.pdbfile.PDBParseError, IndexError):
        # For now, if the requested coordinate set doesn't exist, then we will
        # default to using the only (first) available coordinate set
        struct = pr.parsePDB(pdbid, chain=chid) if use_pdb else pr.parseCIF(
            pdbid, chain=chid)
        if struct and chnum > 1:
            try:
                chain = pr.parsePDB(pdbid, chain=chid, model=1)
                modified_model_number = True
            except:
                return pnid, ERRORS["PARSING_ERROR"]
        else:
            return pnid, ERRORS["PARSING_ERROR"]
    except Exception as e:
        print(e)
        return pnid, ERRORS["UNKNOWN_EXCEPTIONS"]

    if chain is None:
        return pnid, ERRORS["NONE_CHAINS"]

    if modified_model_number:
        return chain, "MODIFIED_MODEL"

    return chain


def get_chain_from_testid(pnid):
    """ Returns a ProDy chain object for a test pnid. Requires local file.

    Args:
        pnid: ProteinNet ID. Must refer to a test-set record.

    Returns:
        ProDy chain object.
    """

    category, caspid = pnid.split("#")
    try:
        chain = pr.parsePDB(
            os.path.join(PROTEINNET_IN_DIR, "targets", caspid + ".pdb"))
    except AttributeError:
        return pnid, ERRORS["TEST_PARSING_ERRORS"]
    try:
        assert chain.numChains() == 1
    except:
        print("Only a single chain should be parsed from the CASP target PDB.")
        return pnid, ERRORS["TEST_PARSING_ERRORS"]
    return chain


def get_chain_from_proteinnetid(pnid, pnid_type):
    """Returns a ProDy chain for a given pnid.

    Args:
        pnid_type:
        pnid: ProteinNet ID

    Returns:
        ProDy chain object.
    """
    # If the ProteinNet ID is from the test set
    if pnid_type == "test":
        chain = get_chain_from_testid(pnid)
    # If the ProteinNet ID is from the train or validation set
    else:
        chain = get_chain_from_trainid(pnid)
    return chain


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
        if no_nans_infs_allzeros(ang) and no_nans_infs_allzeros(coords):
            all_ohs.append(seq)
            all_angs.append(ang)
            all_crds.append(coords)
            all_ids.append(i)
            c += 1
    print(
        f"{(c * 100) / len(results):.1f}% of chains parsed. ({c}/{len(results)})"
    )
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
        assert all([
            l == num_items
            for l in map(len, [data[subset][k] for k in items_recorded])
        ]), f"{subset} lengths don't match."

    for split in VALID_SPLITS:
        valid_len = len(data[f"valid-{split}"]["seq"])
        assert all([l == valid_len for l in map(len, [data[f"valid-{split}"][k] for k in ["ang", "ids", "crd"]])]), \
            "Valid lengths don't match."


def create_data_dict(train_seq, test_seq, train_ang, test_ang, train_crd,
                     test_crd, train_ids, test_ids, all_validation_data):
    """
    Given split data along with the query information that generated it, this function saves the
    data as a Python dictionary, which is then saved to disk using torch.save.
    See commit  d1935a0869720f85c00824f3aecbbfc6b947711c for a method that saves all relevant information.
    """
    # Sort data
    train_ang, train_seq, train_crd, train_ids = sort_data(
        train_ang, train_seq, train_crd, train_ids)
    test_ang, test_seq, test_crd, test_ids = sort_data(test_ang, test_seq,
                                                       test_crd, test_ids)

    # Create a dictionary data structure, using the sin/cos transformed angles
    data = {
        "train": {
            "seq": train_seq,
            "ang": angle_list_to_sin_cos(train_ang),
            "ids": train_ids,
            "crd": train_crd
        },
        "test": {
            "seq": test_seq,
            "ang": angle_list_to_sin_cos(test_ang),
            "ids": test_ids,
            "crd": test_crd
        },
        "settings": {
            "max_len": max(map(len, train_seq + test_seq)),
            "pad_char": GLOBAL_PAD_CHAR
        },
        "description": {f"ProteinNet {CASP_VERSION.upper()}"},
        # To parse date later, use datetime.datetime.strptime(date, "%I:%M%p on %B %d, %Y")
        "date": datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
    }
    max_val_len = 0
    for split, (seq_val, ang_val, crd_val,
                ids_val) in all_validation_data.items():
        ang_val, seq_val, crd_val, ids_val = sort_data(ang_val, seq_val,
                                                       crd_val, ids_val)
        data[f"valid-{split}"] = {}
        data[f"valid-{split}"]["seq"] = seq_val
        data[f"valid-{split}"]["ang"] = angle_list_to_sin_cos(ang_val)
        data[f"valid-{split}"]["crd"] = crd_val
        data[f"valid-{split}"]["ids"] = ids_val
        max_split_len = max(data["settings"]["max_len"], max(map(len, seq_val)))
        max_val_len = max_split_len if max_split_len > max_val_len else max_val_len
    data["settings"]["max_len"] = max(
        list(map(len, train_seq + test_seq)) + [max_val_len])

    data["settings"]["bin-data"] = bin_sequence_data(train_seq,
                                                     maxlen=MAX_SEQ_LEN)
    data["settings"]["angle_means"] = compute_angle_means(data)
    validate_data_dict(data)
    return data


def compute_angle_means(data):
    """ Computes and retuns the mean of the training data angle matrices. """
    train_angles_sincos = np.concatenate(data["train"]["ang"])
    means = np.nanmean(train_angles_sincos, axis=0)
    return means


def add_proteinnetID_to_idx_mapping(data):
    """
    Given an already processes ProteinNet data dictionary, this function adds
    a mapping from ProteinNet ID to the subset and index number where that
    protein can be looked up in the current dictionary. Useful if you'd like
    to quickly extract a certain protein.
    """
    d = {}
    for subset in ["train", "test"
                  ] + [f"valid-{split}" for split in VALID_SPLITS]:
        for idx, pnid in enumerate(data[subset]["ids"]):
            d[pnid] = {"subset": subset, "idx": idx}

    data["pnids"] = d
    return data


def sort_data(angs, seqs, crds, ids):
    """
    Sorts inputs by length, with shortest first.
    """
    sorted_len_indices = [
        a[0] for a in sorted(
            enumerate(angs), key=lambda x: x[1].shape[0], reverse=False)
    ]
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
        args.out_file = "../data/proteinnet/" + CASP_VERSION + "_" + SUFFIX + ".pkl"
    torch.save(data, args.out_file)
    print(f"Data saved to {args.out_file}.")


def main():
    lim = args.limit
    global PN_TRAIN_DICT, PN_VALID_DICT, PN_TEST_DICT
    train_pdb_ids, valid_ids, test_casp_ids = sidechainnet.utils.parse.parse_raw_proteinnet(
        args.input_dir, TRAIN_FILE)
    print("IDs fetched.")
    PN_TRAIN_DICT, PN_VALID_DICT, PN_TEST_DICT = torch.load(
        os.path.join(args.input_dir, "torch", TRAIN_FILE)), torch.load(
            os.path.join(args.input_dir, "torch", "validation.pkl")), torch.load(
                os.path.join(args.input_dir, "torch", "testing.pkl"))
    print(len(train_pdb_ids), len(valid_ids), len(test_casp_ids))
    # Download and preprocess all data from PDB IDs
    with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
        train_results = list(
            tqdm.tqdm(p.imap(process_id, train_pdb_ids[:lim]),
                      total=len(train_pdb_ids[:lim]),
                      dynamic_ncols=True))
    if lim:
        vlim = 1
    else:
        vlim = None
    valid_result_meta = {}
    for split, vids in group_validation_set(valid_ids).items():
        valid_result_meta[split] = []
        for vid in tqdm.tqdm(vids[:vlim], dynamic_ncols=True):
            valid_result_meta[split].append(process_id(vid))

    test_results = []
    for tid in tqdm.tqdm(test_casp_ids[:vlim], dynamic_ncols=True):
        test_results.append(process_id(tid))

    print("Structures processed.")

    # Unpack results
    print("Training set:\t", end="")
    train_ohs, train_angs, train_strs, train_ids = unpack_processed_results(
        train_results, train_pdb_ids[:lim])
    for (split, results), vids in zip(valid_result_meta.items(),
                                      group_validation_set(valid_ids).values()):
        print(f"Valid set {split}%:\t", end="")
        valid_result_meta[split] = unpack_processed_results(
            results, vids[:vlim])
    print("Test set:\t\t", end="")
    test_ohs, test_angs, test_strs, test_ids = unpack_processed_results(
        test_results, test_casp_ids[:vlim])
    ERRORS.summarize()

    # Split into train, test and validation sets. Report sizes.
    data = create_data_dict(train_ohs, test_ohs, train_angs, test_angs,
                            train_strs, test_strs, train_ids, test_ids,
                            valid_result_meta)
    data = add_proteinnetID_to_idx_mapping(data)
    save_data_dict(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Converts a ProteinNet directory of raw records into a dataset for all"
        "atom protein structure prediction.")
    parser.add_argument('input_dir',
                        type=str,
                        help='Path to ProteinNet raw records directory.')
    parser.add_argument('-o',
                        '--out_file',
                        type=str,
                        help='Path to output file (.tch file)')
    parser.add_argument('-l',
                        '--limit',
                        type=int,
                        default=None,
                        help='Limit size of training set for debugging.')
    parser.add_argument("--pdb_dir",
                        default=os.path.expanduser("~/pdb/"),
                        type=str,
                        help="Path for ProDy-downloaded PDB files.")
    parser.add_argument('--training_set',
                        type=int,
                        default=100,
                        help='Which thinning of the training set to parse. '
                        '{30,50,70,90,95,100}. Default 100.')
    args = parser.parse_args()

    TRAIN_FILE = f"training_{args.training_set}.pkl"
    PN_TRAIN_DICT, PN_VALID_DICT, PN_TEST_DICT = None, None, None
    ASTRAL_FILE = "../data/astral_summary.txt"  # combined previous versions of dir.des.scope.2.xx-stable.txt into
    # one big dict
    ASTRAL_ID_MAPPING = parse_astral_summary_file(ASTRAL_FILE)
    SUFFIX = str(
        datetime.datetime.today().strftime("%y%m%d")) + f"_{args.training_set}"
    match = re.search(r"casp\d+", args.input_dir, re.IGNORECASE)
    assert match, "The input_dir is not titled with 'caspX'."
    CASP_VERSION = match.group(0)

    pr.pathPDBFolder(args.pdb_dir)  # Set PDB download location
    np.set_printoptions(
        suppress=True)  # suppresses scientific notation when printing
    np.set_printoptions(threshold=sys.maxsize)  # suppresses '...' when printing

    try:
        main()
    except Exception as e:
        ERRORS.summarize()
        raise e


def load_data(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_data(data, path):
    with open(path, "wb") as f:
        return pickle.dump(data, f)
