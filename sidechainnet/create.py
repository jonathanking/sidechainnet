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
    python create.py $PATH_TO_PROTEINNET_FILES_FOR_SINGLE_CASP --training_set all

To generate a single "thinning" (e.g. 30) for a CASP competition, run:
    python create.py $PATH_TO_PROTEINNET_FILES_FOR_SINGLE_CASP --training_set 30

Author: Jonathan King
Date:   10/28/2020
"""

import argparse
import os
import re
from multiprocessing import Pool, cpu_count

import prody as pr
from tqdm import tqdm

from sidechainnet.utils.align import (assert_mask_gaps_are_correct, expand_data_with_mask,
                                      init_aligner, merge)
from sidechainnet.utils.download import download_sidechain_data
from sidechainnet.utils.errors import write_errors_to_files
from sidechainnet.utils.manual_adjustment import (manually_adjust_data,
                                                  manually_correct_mask,
                                                  needs_manual_adjustment)
from sidechainnet.utils.measure import NUM_COORDS_PER_RES
from sidechainnet.utils.organize import load_data, organize_data, save_data
from sidechainnet.utils.parse import parse_raw_proteinnet

pr.confProDy(verbosity="none")
pr.confProDy(auto_secondary=False)


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

    mask, alignment, ang, crd, dssp, warning = merge(aligner, pn_entry["primary"],
                                                     sc_entry["seq"], sc_entry["ang"],
                                                     sc_entry["crd"], sc_entry["sec"],
                                                     pn_entry["mask"], pnid)
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
        new_entry["msk"] = mask
        new_entry["res"] = sc_entry["res"]

        length = len(pn_entry["primary"])
        for k, v in new_entry.items():
            if k == "crd":
                if len(v) // NUM_COORDS_PER_RES != length:
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


def combine_datasets(proteinnet_out, sc_data, training_set):
    """Adds sidechain information to ProteinNet to create SidechainNet.

    Args:
        proteinnet_out: Location of preprocessed ProteinNet data
        sc_data: Sidechain data dictionary with keys being ProteinNet IDs
        training_set: Which training set thinning to use (i.e. 30, 50,... 100)

    Returns:
        SidechainNet as a dictionary mapping ProteinNet IDs to all data relevant
        for sidechain prediction.
    """
    print("Preparing to merge ProteinNet data with downloaded sidechain data.")
    pn_files = [
        os.path.join(proteinnet_out, f"training_{training_set}.pkl"),
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
    return pndata[pnid], scdata[pnid], pnid


def format_sidechainnet_path(casp_version, training_split):
    """Returns a string representing a .pkl file for a CASP version and training set."""
    if casp_version == "debug":
        return "sidechainnet_debug.pkl"
    return f"sidechainnet_casp{casp_version}_{training_split}.pkl"


def create():
    """Generates SidechainNet for a single CASP thinning."""
    # First, parse raw proteinnet files into Python dictionaries for convenience
    pnids = parse_raw_proteinnet(args.proteinnet_in, args.proteinnet_out,
                                 args.training_set)
    pnids = pnids[:args.limit]  # Limit the length of the list for debugging

    # Using the ProteinNet IDs as a guide, download the relevant sidechain data
    sc_only_data, sc_filename = download_sidechain_data(pnids, args.sidechainnet_out,
                                                        args.casp_version,
                                                        args.training_set, args.limit,
                                                        args.proteinnet_in,
                                                        args.regenerate_scdata)

    # Finally, unify the sidechain data with ProteinNet
    sidechainnet_raw = combine_datasets(args.proteinnet_out, sc_only_data,
                                        args.training_set)

    sidechainnet_outfile = os.path.join(
        args.sidechainnet_out,
        format_sidechainnet_path(args.casp_version, args.training_set))
    sidechainnet = organize_data(sidechainnet_raw, args.proteinnet_out, args.casp_version,
                                 args.training_set)
    save_data(sidechainnet, sidechainnet_outfile)
    print(f"SidechainNet for CASP {args.casp_version} written to {sidechainnet_outfile}.")


def create_all():
    """Generates all thinnings of a particular CASP dataset, starting with the largest."""
    # First, parse raw proteinnet files into Python dictionaries for convenience
    pnids = parse_raw_proteinnet(args.proteinnet_in, args.proteinnet_out, 100)
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
    sidechainnet_raw_100 = combine_datasets(args.proteinnet_out, sc_only_data, 100)

    # Generate debug dataset with 200 training examples
    sc_outfile = os.path.join(args.sidechainnet_out, format_sidechainnet_path("debug", 0))
    debug = organize_data(sidechainnet_raw_100, args.proteinnet_out, "debug", "debug")
    save_data(debug, sc_outfile)
    print(f"SidechainNet for CASP {args.casp_version} (debug) written to {sc_outfile}.")

    # Generate the rest of the training sets
    for training_set in [100, 95, 90, 70, 50, 30]:
        sc_outfile = os.path.join(
            args.sidechainnet_out,
            format_sidechainnet_path(args.casp_version, training_set))
        sidechainnet = organize_data(sidechainnet_raw_100, args.proteinnet_out,
                                     args.casp_version, training_set)
        save_data(sidechainnet, sc_outfile)
        print(f"SidechainNet for CASP {args.casp_version} "
              f"({training_set}% thinning) written to {sc_outfile}.")


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
    parser.add_argument('--training_set',
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
    if args.training_set != 'all':
        args.training_set = int(args.training_set)

    match = re.search(r"casp(\d+)", args.proteinnet_in, re.IGNORECASE)
    if not match:
        raise parser.error("The input_dir does not contain 'caspX'. "
                           "Please ensure the raw files are enclosed "
                           "in a path that contains the CASP version"
                           " i.e. 'casp12'.")
    args.casp_version = match.group(1)

    if args.training_set == 'all':
        create_all()
    else:
        create()
