"""
sidechainnet.py
A protein structure prediction data set that includes side chain information.
A direct extension of ProteinNet by Mohammed AlQuraishi.

"""
import os

import argparse
import itertools
import torch
import multiprocessing
from glob import glob

import sidechainnet
from sidechainnet.parse_proteinnet import process_file


def parse_raw_proteinnet(proteinnet_in_dir, proteinnet_out_dir, training_set):
    """
    Preprocesses raw ProteinNet records by reading them and transforming them
    into a Pytorch-saved dictionary.
    """
    train_file = f"training_{training_set}.pt"
    if os.path.exists(os.path.join(proteinnet_out_dir, train_file)):
        print("Raw ProteinNet files already preprocessed.")
        proteinnet_files = glob(os.path.join(proteinnet_out_dir, "*.pt"))
        proteinnet = {}
        for f in proteinnet_files:
            proteinnet[os.path.basename(f)] = torch.load(f)
        return proteinnet

    # If the torch-preprocessed ProteinNet dictionaries don't exist, create them.
    if not os.path.exists(proteinnet_out_dir):
        os.mkdir(proteinnet_out_dir)

    # Look for the raw ProteinNet files
    input_files = glob(os.path.join(proteinnet_in_dir, "*[!.ids]"))
    assert len(input_files) == 8, f"Looking for raw ProteinNet files in '{proteinnet_in_dir}', but could not find all 8.\n Please download from Mohammed AlQuraishi's repository: https://github.com/aqlaboratory/proteinnet . "

    # Process each ProteinNet file by turning them into PyTorch saved dictionaries
    print("Preprocessing raw ProteinNet files...")
    with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
        p.map(process_file, zip(input_files, itertools.repeat(proteinnet_out_dir)))
    print("Done.")
    return parse_raw_proteinnet(proteinnet_in_dir, proteinnet_out_dir, training_set)


def download_sidechain_data():
    pass


def combine_datasets(sidechains, proteinnet):
    pass


def save_dict(d, path):
    torch.save(d, path)


def save_dataset(dataset, path):
    pass



def main():
    # First, create a Pytorch version of the raw proteinnet files for easier inspection
    proteinnet = parse_raw_proteinnet(args.proteinnet_in, args.proteinnet_out, args.training_set)
    save_dict(proteinnet, args.proteinnet_out)

    # Then, using the proteinnet IDs as a guide, download the relevant sidechain data
    sidechains = download_sidechain_data(proteinnet.keys())
    save_dict(sidechains, args.sidechains_out_file)

    # Finally, combine the raw sidechain data and proteinnet data
    sidechainnet = combine_datasets(sidechains, proteinnet)
    save_dataset(sidechainnet, args.sidechainnet_out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Constructs SidechainNet.")
    parser.add_argument('proteinnet_in', type=str, help='Path to ProteinNet raw records directory.')
    parser.add_argument('--proteinnet_out', '-po', type=str, help='Where to save parsed, raw ProteinNet.', default="../data/proteinnet/")
    parser.add_argument('-o', '--output_dir', type=str, help='Path to output file (.pt file)', default=os.path.curdir)
    parser.add_argument('-l', '--limit', type=int, default=None, help='Limit size of training set for debugging.')
    parser.add_argument("--pdb_dir", default=os.path.expanduser("~/pdb/"), type=str,
                        help="Location to download PDB files for ProDy.")
    parser.add_argument('--training_set', type=int, default=100, help='Which \'thinning\' of the ProteinNet training '
                                                                      'set to parse. {30,50,70,90,95,100}. Default 100.')
    args = parser.parse_args()


    main()
