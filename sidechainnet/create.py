"""
A protein structure prediction data set that includes sidechain information.
A direct extension of ProteinNet by Mohammed AlQuraishi.

"""
import os
import argparse
import re

import prody as pr
pr.confProDy(verbosity="none")

from sidechainnet.download_and_parse import download_sidechain_data, load_data, save_data
from sidechainnet.utils.proteinnet import parse_raw_proteinnet


def combine_datasets(proteinnet_out, sidechainnet_out, training_set):
    pass


def save_full_dataset(train, valid, test, path):
    pass


def main():
    # First, create PyTorch versions of  raw proteinnet files for convenience
    pnids = parse_raw_proteinnet(args.proteinnet_in, args.proteinnet_out,
                                 args.training_set)

    # Using the ProteinNet IDs as a guide, download the relevant sidechain data
    sc_data, sc_filename = download_sidechain_data(pnids, args.sidechainnet_out,
                                                   args.casp_version,
                                                   args.training_set,
                                                   args.limit,
                                                   args.proteinnet_in)

    # Finally, unify the sidechain data with ProteinNet
    combine_datasets(args.proteinnet_out, args.sidechainnet_out, args.training_set)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Constructs SidechainNet.")
    parser.add_argument('proteinnet_in', type=str, help='Path to ProteinNet raw records directory.')
    parser.add_argument('--proteinnet_out', '-po', type=str, help='Where to save parsed, raw ProteinNet.',
                        default="../data/proteinnet/")
    parser.add_argument('--sidechainnet_out', '-so', type=str, help='Where to save SidechainNet.',
                        default="../data/sidechainnet/")
    parser.add_argument('-o', '--output_dir', type=str, help='Path to output file (.pt file)', default=os.path.curdir)
    parser.add_argument('-l', '--limit', type=int, default=None, help='Limit size of training set for debugging.')
    parser.add_argument("--pdb_dir", default=os.path.expanduser("~/pdb/"), type=str,
                        help="Location to download PDB files for ProDy.")
    parser.add_argument('--training_set', type=int, default=100, help='Which \'thinning\' of the ProteinNet training '
                                                                      'set to parse. {30,50,70,90,95,100}. '
                                                                      'Default 100.')
    args = parser.parse_args()

    match = re.search(r"casp\d+", args.proteinnet_in, re.IGNORECASE)
    assert match, "The input_dir is not titled with 'caspX'. Please ensure the raw files are enclosed in a path" \
                  "that contains the CASP version i.e. 'casp12'."
    args.casp_version = match.group(0)

    main()
