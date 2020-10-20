"""
A wrapper for sidechainnet.create.py which generates SidechainNet files for all thinnings
of a given CASP competition. See create.py for more information.

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
from sidechainnet.create import combine_datasets, format_sidechainnet_path


def main():
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
    args = parser.parse_args()

    match = re.search(r"casp(\d+)", args.proteinnet_in, re.IGNORECASE)
    if not match:
        raise argparse.ArgumentError("The input_dir does not contain 'caspX'. "
                                     "Please ensure the raw files are enclosed "
                                     "in a path that contains the CASP version"
                                     " i.e. 'casp12'.")
    args.casp_version = match.group(1)

    main()
