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

from sidechainnet.download_sidechain_data import get_sidechain_data
from sidechainnet.parse_proteinnet import process_file, load_ids_from_text_files


def parse_raw_proteinnet(proteinnet_in_dir, proteinnet_out_dir, training_set):
    """
    Preprocesses raw ProteinNet records by reading them and transforming them
    into Pytorch-saved dictionaries. The files are kept separate due to file size.
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
    assert len(input_files) == 8, f"Looking for raw ProteinNet files in '{proteinnet_in_dir}', but could not find " \
                                   "all 8.\n Please download from Mohammed AlQuraishi's repository: " \
                                   "https://github.com/aqlaboratory/proteinnet"

    # Process each ProteinNet file by turning them into PyTorch saved dictionaries
    print("Preprocessing raw ProteinNet files...")
    with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
        p.map(process_file, zip(input_files, itertools.repeat(proteinnet_out_dir)))
    print("Done.")
    return parse_raw_proteinnet(proteinnet_in_dir, proteinnet_out_dir, training_set)


def download_sidechain_data(proteinnet_out_dir, sidechainnet_out_dir, training_set):
    if not os.path.exists(sidechainnet_out_dir):
        os.mkdir(sidechainnet_out_dir)
    train_ids, valid_ids, test_ids = load_ids_from_text_files(proteinnet_out_dir, training_set)

    train_data = get_sidechain_data(train_ids, os.path.join(proteinnet_out_dir, f"training_{training_set}.pt"),
                                    mode="train")
    valid_data = get_sidechain_data(valid_ids, os.path.join(proteinnet_out_dir, f"validation.pt"), mode="valid")
    test_data = get_sidechain_data(test_ids, os.path.join(proteinnet_out_dir, f"testing.pt"), mode="test")

    return train_data, valid_data, test_data


def combine_datasets(sidechains, proteinnet):
    pass


def save_dict(d, path):
    torch.save(d, path)


def save_full_dataset(train, valid, test, path):
    pass



def main():
    # First, create Pytorch versions of the raw proteinnet files for easier inspection
    parse_raw_proteinnet(args.proteinnet_in, args.proteinnet_out, args.training_set)

    # Then, using the proteinnet IDs as a guide, download the relevant sidechain data
    train, valid, test = download_sidechain_data(args.proteinnet_out, args.sidechainnet_out, args.training_set)
    save_full_dataset(train, valid, test, args.sidechainnet_out)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Constructs SidechainNet.")
    parser.add_argument('proteinnet_in', type=str, help='Path to ProteinNet raw records directory.')
    parser.add_argument('--proteinnet_out', '-po', type=str, help='Where to save parsed, raw ProteinNet.', default="../data/proteinnet/")
    parser.add_argument('--sidechainnet_out', '-so', type=str, help='Where to save SidechainNet.',
                        default="../data/sidechainnet/")
    parser.add_argument('-o', '--output_dir', type=str, help='Path to output file (.pt file)', default=os.path.curdir)
    parser.add_argument('-l', '--limit', type=int, default=None, help='Limit size of training set for debugging.')
    parser.add_argument("--pdb_dir", default=os.path.expanduser("~/pdb/"), type=str,
                        help="Location to download PDB files for ProDy.")
    parser.add_argument('--training_set', type=int, default=100, help='Which \'thinning\' of the ProteinNet training '
                                                                      'set to parse. {30,50,70,90,95,100}. Default 100.')
    args = parser.parse_args()


    main()
