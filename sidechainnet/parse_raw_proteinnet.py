""" This script allows the raw ProteinNet files from
https://github.com/aqlaboratory/proteinnet. """

import os
from glob import glob
import multiprocessing

import torch


def load_ids_from_text_files(directory, train_file):
    """
    Given a directory where raw ProteinNet records are stored along with .ids
    files, reads and returns the contents of those files. Effectively returns
    a list of IDs associated with the training, validation, and test sets.
    """
    with open(os.path.join(directory, train_file.replace(".pt", ".ids")), "r") as trainf, \
            open(os.path.join(directory, "validation.ids"), "r") as validf, \
            open(os.path.join(directory, "testing.ids"), "r") as testf:
        train_ids = trainf.read().splitlines()
        valid_ids = validf.read().splitlines()
        test_ids = testf.read().splitlines()
        return train_ids, valid_ids, test_ids


def read_protein_from_file(file_pointer, include_tertiary):
    """
    Modified from github.com/OpenProtein/openprotein:preprocessing.py on June
    20, 2019. Original carries an MIT license. Copyright (c) 2018 Jeppe
    Hallgren.
    """
    dict_ = {}
    _dssp_dict = {'L': 0, 'H': 1, 'B': 2, 'E': 3, 'G': 4, 'I': 5, 'T': 6, 'S': 7}
    _mask_dict = {'-': 0, '+': 1}

    while True:
        next_line = file_pointer.readline()
        if next_line == '[ID]\n':
            id_ = file_pointer.readline()[:-1]
            dict_.update({'id': id_})
        elif next_line == '[PRIMARY]\n':
            primary = file_pointer.readline()[:-1]
            dict_.update({'primary': primary})
        elif next_line == '[EVOLUTIONARY]\n':
            evolutionary = []
            for residue in range(21): evolutionary.append(
                [float(step) for step in file_pointer.readline().split()])
            dict_.update({'evolutionary': evolutionary})
        elif next_line == '[SECONDARY]\n':
            secondary = list([_dssp_dict[dssp] for dssp in file_pointer.readline()[:-1]])
            dict_.update({'secondary': secondary})
        elif next_line == '[TERTIARY]\n' and include_tertiary:
            tertiary = []
            # 3 dimension
            for axis in range(3): tertiary.append(
                [float(coord) for coord in file_pointer.readline().split()])
            dict_.update({'tertiary': tertiary})
        elif next_line == '[MASK]\n':
            mask = list([_mask_dict[aa] for aa in file_pointer.readline()[:-1]])
            dict_.update({'mask': mask})
        elif next_line == '\n':
            return dict_
        elif next_line == '':
            return None


def process_file(input_filename):
    """
    A parallelizable method for processing one raw ProteinNet file and
    creating a Pytorch-saved python dictionary of the data.
    """
    global torch_dict_dir
    print("    " + input_filename)
    text_file = open(input_filename + '.ids', "w")
    input_file = open(input_filename, "r")
    meta_dict = {}
    while True:
        next_protein = read_protein_from_file(input_file, include_tertiary=False)
        if next_protein is None:
            break
        id_ = next_protein["id"]
        del next_protein["id"]
        meta_dict.update({id_: next_protein})
        text_file.write(f"{id_}\n")
    torch.save(meta_dict, os.path.join(torch_dict_dir, os.path.basename(input_filename) + ".pt"))
    input_file.close()
    text_file.close()
    print(f"{input_filename} finished.")


def parse_raw_proteinnet(proteinnet_dir, training_set):
    """
    Preprocesses raw ProteinNet records by reading them and transforming them
    into a Pytorch-saved dictionary. It excludes the tertiary information as
    this will acquired from the PDB.
    """
    train_file = f"training_{training_set}.pt"
    global torch_dict_dir
    # Test for .pt files existance, return ids and exit if already complete
    torch_dict_dir = os.path.join(proteinnet_dir, "torch/")
    if os.path.exists(os.path.join(torch_dict_dir, train_file)):
        print("Raw ProteinNet files already preprocessed.")
        train_ids, valid_ids, test_ids = load_ids_from_text_files(torch_dict_dir.replace("/torch", "/raw"), train_file)
        return train_ids, valid_ids, test_ids

    # If the torch-preprocessed ProteinNet dictionaries don't exist, create them.
    if not os.path.exists(torch_dict_dir):
        os.mkdir(torch_dict_dir)

    input_files = glob(os.path.join(proteinnet_dir, "raw/*[!.ids]"))
    print("Preprocessing raw ProteinNet files...")

    with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
        p.map(process_file, input_files)
    print("Done.")
    return parse_raw_proteinnet(proteinnet_dir, train_file)