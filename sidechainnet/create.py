"""
A protein structure prediction data set that includes sidechain information.
A direct extension of ProteinNet by Mohammed AlQuraishi.

"""
import argparse
import os
import pickle
import re
from multiprocessing import Pool, cpu_count

import numpy as np
import prody as pr
from tqdm import tqdm

from sidechainnet.utils.align import merge, expand_data_with_mask, assert_mask_gaps_are_correct, binary_mask_to_str

from sidechainnet.utils.errors import write_errors_to_files
from sidechainnet.utils.organize import create_empty_dictionary, validate_data_dict, organize_data, load_data, save_data

pr.confProDy(verbosity="none")

from sidechainnet.utils.download import download_sidechain_data, VALID_SPLITS
from sidechainnet.utils.parse import parse_raw_proteinnet, FULL_ASTRAL_IDS_INCORRECTLY_PARSED
from sidechainnet.utils.align import init_aligner, manually_adjust_data
from sidechainnet.utils.measure import NUM_COORDS_PER_RES, GLOBAL_PAD_CHAR


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
              "As of May 2020, it is not included in ProteinNet.")

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
            print(f"{pnid} had a bad gap with len {bad_gap_len:.2f}.")
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
                assert len(
                    v
                ) // NUM_COORDS_PER_RES == l, f"{k} does not have correct length {l} (is {len(v) // NUM_COORDS_PER_RES})."
            else:
                if len(v) != l:
                    return {}, "failed"
                assert len(v) == l, f"{k} does not have correct length {l} (is {len(v)})."

    return new_entry, warning


def manually_correct_mask(pnid, pn_entry, mask):
    if pnid == "3TDN_1_A":
        # In this case, the default mask from ProteinNet was actually correct. The
        # protein's sequence has two equal scoring alignments, but the aligners typically
        # pick the "incorrect" one.
        mask = binary_mask_to_str(pn_entry['mask'])
    return mask


def needs_manual_adjustment(pnid):
    """Declares a list of pnids that should be handled manually due to eggregious
    differences between observed and expected seqeuences and masks. """
    if pnid in [
            "4PGI_1_A", "3CMG_1_A", "4ARW_1_A", "4Z08_1_A", "2PLV_1_1", "4PG7_1_A",
            "2O24_1_A", "5I4N_1_A", "4RYK_1_A", "1CS4_3_C", "3SRY_1_A", "2AV4_1_A",
            "3GW7_1_A", "1TQ5_1_A", "5DND_1_A", "4YCU_1_A", "1VRZ_1_A", "1RRX_1_A",
            "2XUV_1_A", "2CFO_1_A", "5DNC_1_A", "2WTS_1_A", "4JQI_3_L", "2H9W_1_A",
            "5DNE_1_A", "3RN8_1_A", "4RQF_2_A", "2FLQ_1_A", "3IPN_1_A", "3GP3_1_A",
            "2Q6P_1_A", "4O2D_1_A", "2XXX_1_A", "3AB4_2_B", "2PLV_1_1", "4UQQ_1_A",
            "2DTJ_1_A", "4ORN_1_A", "4PG7_1_A", "2XXR_1_A", "3IG5_1_A", "3FPH_1_A",
            "2O24_1_A", "4RQE_1_A", "2LIG_1_A", "4XMR_1_A", "1CT9_1_A", "1KL1_1_A",
            "3Q1X_1_A", "1II5_1_A", "2XII_1_A", "3SRY_1_A", "4YCW_1_A", "3ZDQ_1_A",
            "1YJS_1_A", "4CVK_1_A", "2VSQ_1_A", "3P47_1_A", "4D57_1_A", "3WVN_1_A",
            "2XXU_1_A", "3VSC_1_A", "3S1T_1_A", "2AV4_1_A", "3RNN_1_A", "1WNU_1_A",
            "4BDL_1_A", "3J9M_79_AY"
    ]:
        return True
    else:
        return False


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

    pn_data, errors = write_errors_to_files(pn_data, results_warnings, sc_data.keys())

    print(f"Finished unifying sidechain information with ProteinNet data.\n"
          f"{len(errors['failed'])} IDs failed to combine successfully.")
    return pn_data


def get_tuple(pndata, scdata, pnid):
    return pndata[pnid], scdata[pnid], pnid


def combine_wrapper(pndata_scdata_pnid):
    pn_data, sc_data, pnid = pndata_scdata_pnid
    aligner = init_aligner()
    return combine(pn_data, sc_data, aligner, pnid)


def main():
    # First, parse raw proteinnet files into Python dictionaries for convenience
    pnids = parse_raw_proteinnet(args.proteinnet_in, args.proteinnet_out,
                                 args.training_set)

    # Using the ProteinNet IDs as a guide, download the relevant sidechain data
    sc_data, sc_filename = download_sidechain_data(pnids, args.sidechainnet_out,
                                                   args.casp_version, args.training_set,
                                                   args.limit, args.proteinnet_in)

    # Finally, unify the sidechain data with ProteinNet
    sidechainnet_raw = combine_datasets(args.proteinnet_out, sc_data, args.training_set)

    sidechainnet_outfile = os.path.join(
        args.sidechainnet_out,
        f"sidechainnet_{args.casp_version}_{args.training_set}.pkl")
    sidechainnet = organize_data(sidechainnet_raw, args.proteinnet_out, args.casp_version)
    save_data(sidechainnet, sidechainnet_outfile)
    print(
        f"SidechainNet for {args.casp_version.upper()} written to {sidechainnet_outfile}."
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
                        type=int,
                        default=30,
                        help='Which \'thinning\' of the ProteinNet training '
                        'set to parse. {30,50,70,90,95,100}. Default 30.')
    args = parser.parse_args()

    match = re.search(r"casp\d+", args.proteinnet_in, re.IGNORECASE)
    if not match:
        raise argparse.ArgumentError("The input_dir does not contain 'caspX'. "
                                     "Please ensure the raw files are enclosed "
                                     "in a path that contains the CASP version"
                                     " i.e. 'casp12'.")
    args.casp_version = match.group(0)

    main()
