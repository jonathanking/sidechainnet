"""
sidechainnet.py
A protein structure prediction data set that includes side chain information. A direct extension of ProteinNet by Mohammed AlQuraishi.

"""
import os

import argparse

import sidechainnet
from sidechainnet.parse_raw_proteinnet import parse_raw_proteinnet


def create_sidechain_dataset():
    pass


def combine_datasets(sidechains, proteinnet):
    pass


def save_dataset(dataset, path):
    pass



def main():

    proteinnet_dataset = parse_raw_proteinnet(args.proteinnet_dir, args.training_set)

    sidechain_dataset = create_sidechain_dataset()

    sidechainnet = combine_datasets(sidechain_dataset, proteinnet_dataset)

    save_dataset(sidechainnet, args.out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Constructs SidechainNet.")
    parser.add_argument('proteinnet_dir', type=str, help='Path to ProteinNet raw records directory.')
    parser.add_argument('-o', '--output_dir', type=str, help='Path to output file (.tch file)', default=os.path.curdir)
    parser.add_argument('-l', '--limit', type=int, default=None, help='Limit size of training set for debugging.')
    parser.add_argument("--pdb_dir", default=os.path.expanduser("~/pdb/"), type=str,
                        help="Location to download PDB files for ProDy.")
    parser.add_argument('--training_set', type=int, default=100, help='Which \'thinning\' of the ProteinNet training '
                                                                      'set to parse. {30,50,70,90,95,100}. Default 100.')
    args = parser.parse_args()
