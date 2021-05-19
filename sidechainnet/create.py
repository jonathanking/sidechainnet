"""Generates SidechainNet files.

SidechainNet is an all-atom protein structure prediction dataset for machine learning.
SidechainNet extends ProteinNet, including angles and coordinates that describe both
the backbone and sidechain components of each protein structure.

The procedure is as follows:
    0. Download ProteinNet raw/text files before calling this script.
    1. Parse raw/text ProteinNet files into Python dictionaries.
    2. Take ProteinNet IDs (pnids) and download the corresponding all-atom information.
    3. Unify the data provided by ProteinNet with the all-atom data by aligning the
        protein sequences from ProteinNet with those observed during downloading.

To generate all ProteinNet datasets for a CASP competition, run:
    python create.py $PATH_TO_PROTEINNET_FILES_FOR_SINGLE_CASP --thinning all

To generate a single "thinning" (e.g. 30) for a CASP competition, run:
    python create.py $PATH_TO_PROTEINNET_FILES_FOR_SINGLE_CASP --thinning 30

Author: Jonathan King
Date:   10/28/2020
"""

import argparse
from collections import namedtuple
import os
from multiprocessing import Pool, cpu_count
import pkg_resources
from sidechainnet.utils.sequence import ONE_TO_THREE_LETTER_MAP

import numpy as np
import prody as pr
from tqdm import tqdm

from sidechainnet.utils.align import (assert_mask_gaps_are_correct, expand_data_with_mask,
                                      init_aligner, merge)
from sidechainnet.utils.download import PN_VALID_SPLITS, _reinit_global_valid_splits, download_complete_proteinnet, download_sidechain_data, get_sequence_from_pnid
from sidechainnet.utils.errors import write_errors_to_files
from sidechainnet.utils.manual_adjustment import (manually_adjust_data,
                                                  manually_correct_mask,
                                                  needs_manual_adjustment)
from sidechainnet.utils.measure import NUM_COORDS_PER_RES
from sidechainnet.utils.organize import get_validation_split_identifiers_from_pnid_list, load_data, organize_data, save_data
from sidechainnet.utils.parse import parse_raw_proteinnet

PNID_CSV_FILE = None

pr.confProDy(verbosity="none")
pr.confProDy(auto_secondary=False)

ArgsTuple = namedtuple(
    "ArgsTuple", "casp_version thinning proteinnet_in proteinnet_out "
    "sidechainnet_out regenerate_scdata limit")


def combine(pn_entry, sc_entry, aligner, pnid):
    """Supplements one entry in ProteinNet with sidechain information.

    Args:
        aligner: A sequence aligner with desired settings. See
            utils.alignment.init_aligner().
        pn_entry: A dictionary describing a single ProteinNet protein. Contains
            sequence, coordinates, PSSMs, secondary structure.
        sc_entry: A dictionary describing the sidechain information for the same
            protein. Contains sequence, coordinates, and angles.

    Returns:
        A dictionary that has unified the two sets of information.
    """

    sc_entry = manually_adjust_data(pnid, sc_entry)
    if needs_manual_adjustment(pnid):
        return {}, "needs manual adjustment"

    # If there is no corresponding ProteinNet entry, we create a template entry
    if pn_entry is None:
        seq = get_sequence_from_pnid(pnid)
        pn_entry = {
            "primary": seq,
            "evolutionary": np.zeros((len(seq), 21)),
            "mask": "?" * len(seq)
        }
        alignment = True
        crd = sc_entry['crd']
        ang = sc_entry['ang']
        dssp = sc_entry['sec']
        ignore_pnmask = True
    else:
        ignore_pnmask = False

    mask, alignment, ang, crd, dssp, unmod_seq, is_mod, warning = merge(
        aligner, pn_entry, sc_entry, pnid, ignore_pnmask=ignore_pnmask)

    new_entry = {}

    if alignment:
        # Create new SidechainNet entry containing all information
        new_entry["seq"] = pn_entry["primary"]
        new_entry["evo"] = pn_entry["evolutionary"]

        correct_gaps, bad_gap_len = assert_mask_gaps_are_correct(mask, crd)
        if not correct_gaps:
            return {}, "bad gaps"

        # We may need to add padding where specified by the mask
        mask = manually_correct_mask(pnid, pn_entry, mask)
        new_entry["ang"] = expand_data_with_mask(ang, mask)
        new_entry["crd"] = expand_data_with_mask(crd, mask)
        new_entry["sec"] = expand_data_with_mask(dssp, mask)
        new_entry["ums"] = make_unmodified_seq_entry(new_entry["seq"], unmod_seq, mask)
        new_entry["mod"] = expand_data_with_mask(is_mod, mask)
        new_entry["msk"] = mask
        new_entry["res"] = sc_entry["res"]

        length = len(pn_entry["primary"])
        for k, v in new_entry.items():
            if k == "crd":
                if len(v) // NUM_COORDS_PER_RES != length:
                    return {}, "failed"
            elif k == "ums":
                if len(v.split(" ")) != length:
                    return {}, "failed"
            elif k != "res":
                if len(v) != length:
                    return {}, "failed"

    return new_entry, warning


def combine_wrapper(pndata_scdata_pnid):
    """Wraps a call to combine for use with multiprocessing.pool."""
    pn_data, sc_data, pnid = pndata_scdata_pnid
    aligner = init_aligner()
    return combine(pn_data, sc_data, aligner, pnid)


def make_unmodified_seq_entry(pn_seq, unmod_seq, mask):
    """Given observed residues, create the unmodified sequence entry for SidechainNet."""
    padded_unmod_seq = expand_data_with_mask(unmod_seq, mask)
    unmod_seq_complete = []
    for c_pn, c_unmod in zip(pn_seq, padded_unmod_seq):
        if c_unmod == "---":
            unmod_seq_complete.append(ONE_TO_THREE_LETTER_MAP[c_pn])
        else:
            unmod_seq_complete.append(c_unmod)
    return " ".join(unmod_seq_complete)


def combine_datasets(proteinnet_out, sc_data, thinning=100):
    """Adds sidechain information to ProteinNet to create SidechainNet.

    Args:
        proteinnet_out: Location of preprocessed ProteinNet data
        sc_data: Sidechain data dictionary with keys being ProteinNet IDs
        thinning: Which training set thinning to use (i.e. 30, 50,... 100)

    Returns:
        SidechainNet as a dictionary mapping ProteinNet IDs to all data relevant
        for sidechain prediction.
    """
    print("Preparing to merge ProteinNet data with downloaded sidechain data.")
    pn_files = [
        os.path.join(proteinnet_out, f"training_{thinning}.pkl"),
        os.path.join(proteinnet_out, "validation.pkl"),
        os.path.join(proteinnet_out, "testing.pkl")
    ]

    pn_data = {}
    for f in pn_files:
        d = load_data(f)
        pn_data.update(d)
    del d

    with Pool(cpu_count()) as p:
        tuples = (get_tuple(pn_data, sc_data, pnid) for pnid in sc_data.keys())
        results_warnings = list(
            tqdm(p.imap(combine_wrapper, tuples),
                 total=len(sc_data.keys()),
                 dynamic_ncols=True))

    combined_data, errors = write_errors_to_files(results_warnings, sc_data.keys())

    print(f"Finished unifying sidechain information with ProteinNet data.\n"
          f"{len(errors['failed'])} IDs failed to combine successfully.")
    return combined_data


def get_tuple(pndata, scdata, pnid):
    """Extract relevant SidechainNet and ProteinNet data from their respective dicts."""
    try:
        return pndata[pnid], scdata[pnid], pnid
    except KeyError:
        return None, scdata[pnid], pnid


def format_sidechainnet_path(casp_version, training_split):
    """Returns a string representing a .pkl file for a CASP version and training set."""
    if casp_version == "debug":
        return "sidechainnet_debug.pkl"
    return f"sidechainnet_casp{casp_version}_{training_split}.pkl"


def create(casp_version=12,
           thinning=30,
           sidechainnet_out="./sidechainnet_data",
           regenerate_scdata=False,
           limit=None):
    """Generate the requested SidechainNet dataset and save pickled result files.

    This function replicates CLI behavior of calling `python sidechainnet/create.py`.

    Args:
        casp_version (int, optional): CASP dataset version (7-12). Defaults to 12.
        thinning (int, optional): Training set thinning (30, 50, 70, 90, 95, 100
            where 100 means 100% of the training set is kept). If 'all', generate all
            training set thinnings. Defaults to 30.
        sidechainnet_out (str, optional): Path for saving processed SidechainNet records.
            Defaults to "data/sidechainnet/".
        regenerate_scdata (bool, optional): If true, regenerate raw sidechain-applicable
            data instead of searching for data that has already been preprocessed.
            Defaults to False.
        limit (bool, optional): The upper limit on number of proteins to process,
            useful when debugging. Defaults to None.

    Raises:
        ValueError: when ProteinNet data paths are non-existant or not as expected.
    """
    if casp_version == "debug":
        raise ValueError("'debug' is not currently supported by scn.create.\n"
                         "Use scn.create(12, 'all') and a debug dataset will be created.")

    # Download ProteinNet custom-helper package (concatenated ProteinNet datasets)
    proteinnet_in = download_complete_proteinnet()
    proteinnet_out = proteinnet_in

    args = ArgsTuple(casp_version, thinning, proteinnet_in, proteinnet_out,
                     sidechainnet_out, regenerate_scdata, limit)
    main(args)


def _create(args):
    """Generates SidechainNet for a single CASP thinning."""
    # First, parse raw proteinnet files into Python dictionaries for convenience
    pnids = get_proteinnet_ids(casp_version=args.casp_version,
                               split="all",
                               thinning=args.thinning)
    pnids = pnids[:args.limit]  # Limit the length of the list for debugging

    # Using the ProteinNet IDs as a guide, download the relevant sidechain data
    sc_only_data, sc_filename = download_sidechain_data(pnids, args.sidechainnet_out,
                                                        args.casp_version, args.thinning,
                                                        args.limit, args.proteinnet_in,
                                                        args.regenerate_scdata)

    # Finally, unify the sidechain data with ProteinNet
    sidechainnet_raw = combine_datasets(args.proteinnet_out, sc_only_data)

    sidechainnet_outfile = os.path.join(
        args.sidechainnet_out, format_sidechainnet_path(args.casp_version, args.thinning))
    sidechainnet = organize_data(sidechainnet_raw, args.casp_version, args.thinning)
    save_data(sidechainnet, sidechainnet_outfile)
    print(f"SidechainNet for CASP {args.casp_version} written to {sidechainnet_outfile}.")


def _create_all(args):
    """Generate all thinnings of a particular CASP dataset, starting with the largest."""
    # First, parse raw proteinnet files into Python dictionaries for convenience
    pnids = get_proteinnet_ids(casp_version=args.casp_version, split="all", thinning=100)
    pnids = pnids[:args.limit]  # Limit the length of the list for debugging

    # Using the ProteinNet IDs as a guide, download the relevant sidechain data
    sc_only_data, sc_filename = download_sidechain_data(
        pnids,
        args.sidechainnet_out,
        args.casp_version,
        100,
        args.limit,
        args.proteinnet_in,
        regenerate_scdata=args.regenerate_scdata)

    # Finally, unify the sidechain data with ProteinNet
    sidechainnet_raw_100 = combine_datasets(args.proteinnet_out, sc_only_data)

    # Generate debug dataset with 200 training examples
    sc_outfile = os.path.join(args.sidechainnet_out, format_sidechainnet_path("debug", 0))
    debug = organize_data(sidechainnet_raw_100,
                          args.casp_version,
                          thinning=100,
                          is_debug=True)
    save_data(debug, sc_outfile)
    print(f"SidechainNet for CASP {args.casp_version} (debug) written to {sc_outfile}.")

    # Generate the rest of the training sets
    for thinning in [100, 95, 90, 70, 50, 30]:
        sc_outfile = os.path.join(args.sidechainnet_out,
                                  format_sidechainnet_path(args.casp_version, thinning))
        sidechainnet = organize_data(sidechainnet_raw_100, args.casp_version, thinning)
        save_data(sidechainnet, sc_outfile)
        print(f"SidechainNet for CASP {args.casp_version} "
              f"({thinning}% thinning) written to {sc_outfile}.")


def create_custom(pnids,
                  output_filename,
                  sidechainnet_out="./sidechainnet_data",
                  short_description="Custom SidechainNet dataset.",
                  regenerate_scdata=False):
    """Generate a custom SidechainNet dataset from user-specified ProteinNet IDs.

    This function utilizes a concatedated version of ProteinNet generated by the author.
    This dataset contains the 100% training set thinning from CASP 12, as well as the
    concatenation of every testing and validation sets from CASPs 7-12. By collecting
    this information into one directory (which this function downloads), the user can
    specify any set of ProteinNet IDs that they would like to include, and this
    function will be abel to access such data if it is available.

    Args:
        pnids (List): List of ProteinNet-formatted protein identifiers (i.e., formmated
            according to <class>#<pdb_id>_<chain_number>_<chain_id>. ASTRAL identifiers
            are also supported, <class>#<pdb_id>_<ASTRAL_id>.)
        output_filename (str): Path to save custom dataset (a pickled Python
            dictionary). ".pkl" extension is recommended.
        sidechainnet_out (str, optional): Path to save processed SidechainNet data.
            Defaults to "data/sidechainnet/".
        short_description (str, optional): A short description provided by the user to
            describe the dataset. Defaults to "Custom SidechainNet dataset.".
        regenerate_scdata (bool, optional): If true, regenerate raw sidechain-applicable
            data instead of searching for data that has already been preprocessed.
            Defaults to False.

    Returns:
        dict: Saves and returns the requested custom SidechainNet dictionary.
    """
    # Download ProteinNet custom-helper package (concatenated ProteinNet datasets)
    proteinnet_in = proteinnet_out = download_complete_proteinnet()

    # Initialize DSSP data
    from sidechainnet.utils.download import _init_dssp_data
    _init_dssp_data()
    new_splits = get_validation_split_identifiers_from_pnid_list(pnids)
    _reinit_global_valid_splits(new_splits)

    # First, parse and load raw proteinnet files into Python dictionaries for convenience
    print(f"Loading complete ProteinNet data (100% thinning) from {proteinnet_in}.")
    _ = parse_raw_proteinnet(proteinnet_in,
                             proteinnet_out,
                             thinning=100,
                             remove_raw_proteinnet=True)

    # Download and return requested pnids
    print("Preparing to download requested proteins via their ProteinNet IDs.")
    dirs, tail = os.path.split(output_filename)
    intermediate_filename = os.path.join(dirs, "sidechain-only_" + tail)
    sc_only_data, sc_filename = download_sidechain_data(
        pnids,
        sidechainnet_out,
        casp_version=None,
        thinning=None,
        limit=None,
        proteinnet_in=proteinnet_in,
        regenerate_scdata=regenerate_scdata,
        output_name=intermediate_filename)

    # Finally, unify the sidechain data with ProteinNet
    sidechainnet_raw = combine_datasets(proteinnet_out, sc_only_data)

    sidechainnet_outfile = os.path.join(sidechainnet_out, output_filename)
    sidechainnet_dict = organize_data(sidechainnet_raw,
                                      casp_version="User-specified",
                                      thinning="User-specified",
                                      description=short_description,
                                      custom_ids=pnids)
    save_data(sidechainnet_dict, sidechainnet_outfile)
    print(
        f"User-specified SidechainNet written to {sidechainnet_outfile}.\n"
        "To load the data in a different format, use sidechainnet.load with the desired\n"
        f"options and set 'local_scn_path={sidechainnet_outfile}'.")

    return sidechainnet_dict


def get_proteinnet_ids(casp_version, split, thinning=None):
    """Return a list of ProteinNet IDs for a given CASP version, split, and thinning.

    Args:
        casp_version (int): CASP version (7, 8, 9, 10, 11, 12).
        split (string): Dataset split ('train', 'valid', 'test'). Validation sets may
            also be specified, ('valid-10', 'valid-20, 'valid-30', 'valid-40', 
            'valid-50', 'valid-70', 'valid-90'). If no valid split is specified, all
            validation set splits will be returned. If split == 'all', the training,
            validation, and testing set splits for the specified CASP and training set
            thinning are all returned.
        thinning (int): Training dataset split thinning (30, 50, 70, 90, 95, 100). Default
            None.

    Returns:
        List: Python list of strings representing the ProteinNet IDs in the requested
            split.
    """
    # Load ProteinNet CSV
    import pandas
    global PNID_CSV_FILE
    if PNID_CSV_FILE is None:
        PNID_CSV_FILE = pandas.read_csv(
            pkg_resources.resource_filename(
                "sidechainnet",
                "resources/all_proteinnet_ids.csv")).set_index('pnid').astype(bool)

    # Fix input, select correct column name for the CSV
    validsplitnum = None
    if "valid" in split:
        if split not in PN_VALID_SPLITS + ['valid']:
            raise ValueError(f"{split} is not a valid ProteinNet validation set split. "
                             f"Use one of {PN_VALID_SPLITS + ['valid']}.")
        if split != "valid":
            _, validsplitnum = split.split("-")
        split = "validation"
    elif split == "train":
        split = "training"
    elif split == "test":
        split = "testing"
    if split in ["all", "training"] and thinning is None:
        raise ValueError("Training set thinning must not be None.")

    def make_colname(cur_split):
        """Return column name for a given CASP dataset split and thinning."""
        colname = f"casp{casp_version}_{cur_split}"
        if cur_split == 'training':
            colname += f"_{thinning}"
        return colname

    # Pick out the pnids that match the requested splits
    if split == "all":
        all_ids = []
        for s in ["training", "validation", "testing"]:
            colname = make_colname(s)
            all_ids.extend(list(PNID_CSV_FILE[PNID_CSV_FILE[colname]].index.values))
        return all_ids

    else:
        colname = make_colname(split)
        if split == "validation" and validsplitnum is not None:
            return list(
                filter(lambda x: x.startswith(f"{validsplitnum}#"),
                       PNID_CSV_FILE[PNID_CSV_FILE[colname]].index.values))
        return list(PNID_CSV_FILE[PNID_CSV_FILE[colname]].index.values)


def generate_all():
    """Generate all SidechainNet datasets for curation and upload."""
    import time
    import sidechainnet as scn
    t = time.localtime()
    timestamp = time.strftime('%b-%d-%Y-%H%M', t)
    pr.startLogfile(f"sidechainnet_generateall_{timestamp}")
    casps = list(range(7, 13))[::-1]
    for c in casps:
        print("CASP", c)
        scn.create(c, "all", regenerate_scdata=False)


def main(args_tuple):
    """Run _create or _create_all using the arguments provided by the namedtuple."""
    if args_tuple.thinning != 'all':
        args_tuple = args_tuple._replace(thinning=int(args_tuple.thinning))

    # Initialize DSSP data
    from sidechainnet.utils.download import _init_dssp_data
    _init_dssp_data()

    if args_tuple.thinning == 'all':
        _create_all(args_tuple)
    else:
        _create(args_tuple)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Constructs SidechainNet.")
    parser.add_argument('proteinnet_in',
                        type=str,
                        help='Path to ProteinNet raw records directory.')
    parser.add_argument('--proteinnet_out',
                        '-po',
                        type=str,
                        help='Where to save parsed, raw ProteinNet.',
                        default="../data/proteinnet/")
    parser.add_argument('--sidechainnet_out',
                        '-so',
                        type=str,
                        help='Where to save SidechainNet.',
                        default="../data/sidechainnet/")
    parser.add_argument('-l',
                        '--limit',
                        type=int,
                        default=None,
                        help='Limit size of training set for debugging.')
    parser.add_argument("--pdb_dir",
                        default=os.path.expanduser("~/pdb/"),
                        type=str,
                        help="Location to download PDB files for ProDy.")
    parser.add_argument('--thinning',
                        type=str,
                        default='30',
                        help='Which \'thinning\' of the ProteinNet training '
                        'set to parse. {30,50,70,90,95,100}. Default 30.')
    parser.add_argument(
        '--regenerate_scdata',
        action="store_true",
        help=('If True, then regenerate the sidechain-only data even if it already exists'
              ' locally.'))
    args = parser.parse_args()
    args_tuple = ArgsTuple(args.casp_version, args.thinning, args.proteinnet_in,
                           args.proteinnet_out, args.sidechainnet_out,
                           args.regenerate_scdata, args.limit)
    main(args_tuple)
