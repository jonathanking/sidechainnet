"""
A protein structure prediction data set that includes sidechain information.
A direct extension of ProteinNet by Mohammed AlQuraishi.

"""
import argparse
import os
import re
from multiprocessing import Pool, cpu_count

import prody as pr
from tqdm import tqdm

from sidechainnet.utils.align import merge, expand_data_with_mask, assert_mask_gaps_are_correct

from sidechainnet.utils.errors import write_errors_to_files
from sidechainnet.utils.manual_adjustment import manually_correct_mask, \
    needs_manual_adjustment
from sidechainnet.utils.organize import organize_data, load_data, save_data

pr.confProDy(verbosity="none")

from sidechainnet.utils.download import download_sidechain_data
from sidechainnet.utils.parse import parse_raw_proteinnet
from sidechainnet.utils.align import init_aligner, manually_adjust_data
from sidechainnet.utils.measure import NUM_COORDS_PER_RES


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
    if "secondary" in pn_entry:
        print("WARNING: secondary structure information is not yet supported. "
              "As of October 2020, it is not included in ProteinNet.")

    sc_entry = manually_adjust_data(pnid, sc_entry)
    if needs_manual_adjustment(pnid):
        return {}, "needs manual adjustment"

    mask, alignment, ang, crd, warning = merge(aligner, pn_entry["primary"],
                                               sc_entry["seq"], sc_entry["ang"],
                                               sc_entry["crd"], pn_entry["mask"], pnid)
    new_entry = {}

    if alignment:
        # Create new SidechainNet entry containing all information
        new_entry["seq"] = pn_entry["primary"]
        new_entry["evo"] = pn_entry["evolutionary"]

        correct_gaps, bad_gap_len = assert_mask_gaps_are_correct(mask, crd, pnid)
        if not correct_gaps:
            return {}, "bad gaps"

        # We may need to add padding where specified by the mask
        mask = manually_correct_mask(pnid, pn_entry, mask)
        new_entry["ang"] = expand_data_with_mask(ang, mask)
        new_entry["crd"] = expand_data_with_mask(crd, mask)
        new_entry["msk"] = mask

        l = len(pn_entry["primary"])
        for k, v in new_entry.items():
            if k == "crd":
                if len(v) // NUM_COORDS_PER_RES != l:
                    return {}, "failed"
            else:
                if len(v) != l:
                    return {}, "failed"
                assert len(v) == l, f"{k} does not have correct length {l} (is {len(v)})."

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
        os.path.join(proteinnet_out, f"validation.pkl"),
        os.path.join(proteinnet_out, f"testing.pkl")
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
    print(
        f"SidechainNet for CASP {args.casp_version} written to {sidechainnet_outfile}."
    )


def create_all():
    """Generates all thinnings of a particular CASP dataset, starting with the largest."""
    # First, parse raw proteinnet files into Python dictionaries for convenience
    pnids = parse_raw_proteinnet(args.proteinnet_in, args.proteinnet_out, 100)
    pnids = pnids[:args.limit]  # Limit the length of the list for debugging

    # Using the ProteinNet IDs as a guide, download the relevant sidechain data
    sc_only_data, sc_filename = download_sidechain_data(pnids,
                                                        args.sidechainnet_out,
                                                        args.casp_version,
                                                        100,
                                                        args.limit,
                                                        args.proteinnet_in,
                                                        regenerate_scdata=True)

    # Finally, unify the sidechain data with ProteinNet
    sidechainnet_raw_100 = combine_datasets(args.proteinnet_out, sc_only_data, 100)

    for training_set in [100, 95, 90, 70, 50, 30]:
        sc_outfile = os.path.join(
            args.sidechainnet_out,
            format_sidechainnet_path(args.casp_version, training_set))
        sidechainnet = organize_data(sidechainnet_raw_100, args.proteinnet_out,
                                     args.casp_version, training_set)
        save_data(sidechainnet, sc_outfile)
        print(f"SidechainNet for CASP {args.casp_version} "
              f"({training_set}% thinning) written to {sc_outfile}."
        )
        

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
        raise argparse.ArgumentError("The input_dir does not contain 'caspX'. "
                                     "Please ensure the raw files are enclosed "
                                     "in a path that contains the CASP version"
                                     " i.e. 'casp12'.")
    args.casp_version = match.group(1)

    if args.training_set == 'all':
        create_all()
    else:
        create()
