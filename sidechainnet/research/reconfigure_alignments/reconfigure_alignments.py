"""This script will prepare MSAs for the SideChainNet dataset given RODA alignments.

Because SidechainNet is not strictly organized by PDBID_CHAINID and this is the format
expected in the RODA alignment database constructed by OpenFold, we create a new alignment
directory for SidechainNet organized by SidechainNet ID. This involves identifying the
correct PDBID_CHAINID directory for each SidechainNet ID and modifying the alignment file
as needed.
"""
from functools import reduce
import glob
import re
import shutil
import string
from typing import Sequence, Tuple
import sidechainnet as scn
import os
import Bio.SeqIO
import A3MIO
import pickle
from tqdm import tqdm
from Bio import Align
from Bio.Align import substitution_matrices

from sidechainnet.utils.download import get_pdbid_from_pnid
from sidechainnet.utils.align import (get_mask_from_alignment,
                                      get_padded_second_seq_from_alignment,
                                      get_padded_first_seq_from_alignment)

DeletionMatrix = Sequence[Sequence[int]]


def make_new_alignment_dir(new_alignment_dir, scn_id):
    os.makedirs(os.path.join(new_alignment_dir, scn_id), exist_ok=True)


def get_roda_dirname(roda_alignment_dir, scn_id):
    try:
        pdb_id, chain, is_astral = get_pdbid_from_pnid(scn_id,
                                                       return_chain=True,
                                                       include_is_astral=True)
    except ValueError:
        return None
    roda_dirname = os.path.join(roda_alignment_dir, f"{pdb_id.lower()}_{chain}")
    return roda_dirname


def load_alignment_via_openfold(filename):
    # records = Bio.SeqIO.parse(filename, "a3m")
    # alignment = Bio.Align.MultipleSeqAlignment(records)
    with open(filename, "r") as f:
        a3mfilestr = f.read()
    alignment, delmatrix = parse_a3m(a3mfilestr)
    return alignment


def load_alignment_with_bio(filename):
    records = Bio.SeqIO.parse(filename, "a3m")
    alignment = Bio.Align.MultipleSeqAlignment(records)
    return alignment


def get_query_sequence(filename):
    with open(filename, "r") as f:
        _ = f.readline()
        query_seq = f.readline().rstrip('\n')
    return query_seq


def check_if_scnseq_is_identical_to_query_seq(query_seq, p):
    if p.seq == query_seq:
        return "True"
    elif re.compile(query_seq.replace("X", ".")).search(p.seq) is not None:
        return "True"
    elif p.seq in query_seq:
        return "Subsequence"
    else:
        return "False"


# def trim_alignment(alignment_name, p):
#     alignment = load_alignment_with_bio(alignment_name)
#     # Find the first and last non-gap characters in the query sequence
#     first_non_gap = re.search(r"[A-Z]", p.seq).start()
#     last_non_gap = re.search(r"[A-Z]", p.seq[::-1]).start()
#     # Trim the alignment to only include the non-gap characters
#     trimmed_alignment = alignment[:, first_non_gap:-last_non_gap]
#     return trimmed_alignment


# From Openfold
def parse_a3m(a3m_string: str) -> Tuple[Sequence[str], DeletionMatrix]:
    """Parses sequences and deletion matrix from a3m format alignment.

    Args:
        a3m_string: The string contents of a a3m file. The first sequence in the
            file should be the query sequence.

    Returns:
        A tuple of:
            * A list of sequences that have been aligned to the query. These
                might contain duplicates.
            * The deletion matrix for the alignment as a list of lists. The element
                at `deletion_matrix[i][j]` is the number of residues deleted from
                the aligned sequence i at residue position j.
    """
    sequences, _ = parse_fasta(a3m_string)
    deletion_matrix = []
    for msa_sequence in sequences:
        deletion_vec = []
        deletion_count = 0
        for j in msa_sequence:
            if j.islower():
                deletion_count += 1
            else:
                deletion_vec.append(deletion_count)
                deletion_count = 0
        deletion_matrix.append(deletion_vec)

    # Make the MSA matrix out of aligned (deletion-free) sequences.
    deletion_table = str.maketrans("", "", string.ascii_lowercase)
    aligned_sequences = [s.translate(deletion_table) for s in sequences]
    return aligned_sequences, deletion_matrix


def parse_fasta(fasta_string: str) -> Tuple[Sequence[str], Sequence[str]]:
    """Parses FASTA string and returns list of strings with amino-acid sequences.

    Arguments:
        fasta_string: The string contents of a FASTA file.

    Returns:
        A tuple of two lists:
        * A list of sequences.
        * A list of sequence descriptions taken from the comment lines. In the
            same order as the sequences.
    """
    sequences = []
    descriptions = []
    index = -1
    for line in fasta_string.splitlines():
        line = line.strip()
        if line.startswith(">"):
            index += 1
            descriptions.append(line[1:])  # Remove the '>' at the beginning.
            sequences.append("")
            continue
        elif line.startswith("#"):
            continue
        elif not line:
            continue  # Skip blank lines.
        sequences[index] += line

    return sequences, descriptions


def align_and_select_cols(alignment_name, p):
    """Align the SCN seq to the MSA and select the columns that match the query sequence.

    This is necessary because the RODA alignments are not necessarily aligned to the query sequence.
    """
    # Load the alignment
    alignment = load_alignment_with_bio(alignment_name)
    query = get_query_sequence(alignment_name)
    # Align the SCN sequence to the MSA query sequence
    aligner = get_aligner()
    my_alignment = next(aligner.align(query, p.seq))
    padded_query = get_padded_first_seq_from_alignment(my_alignment)
    padded_mask = get_mask_from_alignment(my_alignment)
    padded_scnseq = get_padded_second_seq_from_alignment(my_alignment)

    selected_columns = []
    j = 0
    for i, (q, m, s) in enumerate(zip(padded_query, padded_mask, padded_scnseq)):
        if m == "+":
            selected_columns.append(j)
        if q != "-" or (q == "-" and alignment[0, j] == "-"):
            j += 1
    sub_alignment_columns = (alignment[:, col:col+1] for col in selected_columns)
    selected_alignment = reduce(lambda x, y: x + y, sub_alignment_columns)
    return selected_alignment

    # # Select the columns that match the query sequence
    # selected_columns = []
    # for i in range(len(alignment[0])):
    #     if alignment[0][i] == alignment[1][i]:
    #         selected_columns.append(i)
    # # Trim the alignment to only include the selected columns
    # trimmed_alignment = alignment[:, selected_columns]
    # return trimmed_alignment


def get_aligner():
    da = Align.PairwiseAligner()
    da.substitution_matrix = substitution_matrices.load("BLOSUM62")
    # Prefer gaps on ends of sequences
    da.end_gap_score = -1
    da.internal_gap_score = -2
    # Prefer to extend rather than open
    da.internal_open_gap_score = -1.9
    da.internal_extend_gap_score = -1.5
    da.match_score = 3
    da.mismatch_score = -.1
    da.wildcard = "X"
    return da


def write_new_alignment(alignment, output_filename, fmt="clustal"):
    """Write the new alignment to a file."""
    with open(output_filename, "w") as f:
        f.write(format(alignment, fmt))


def write_new_query_sequence_to_fasta_file(input_fasta_file, p):
    """Write the new query sequence to a fasta file."""
    # Get the new query sequence
    new_query_sequence = p.seq
    # Write the new query sequence to a fasta file
    input_fasta_file.write(f">{p.id}\n{new_query_sequence}\n")
    input_fasta_file.flush()


def main(roda_alignment_dir, new_alignment_dir):
    # identical_file = open(
    #     "/home/jok120/sidechainnet/sidechainnet/research/reconfigure_alignments/"
    #     "identical.txt", "w")
    # not_identifcal_file = open(
    #     "/home/jok120/sidechainnet/sidechainnet/research/reconfigure_alignments/"
    #     "not_identical.txt", "w")
    # subsequence_file = open(
    #     "/home/jok120/sidechainnet/sidechainnet/research/reconfigure_alignments/"
    #     "subsequence.txt", "w")
    # unreported_gap_file = open(
    #     "/home/jok120/sidechainnet/sidechainnet/research/reconfigure_alignments/"
    #     "unreported_gap.txt", "w")
    # input_fasta_file = open("/scr/experiments/221101/input.fasta", "w")
    d = scn.load(local_scn_path="/home/jok120/scn221001/sidechainnet_casp12_100.pkl",
                 trim_edges=False)

    # Load the minimized SidechainNet
    scn_minimized = scn.load(local_scn_path="/home/jok120/scnmin221013/scn_minimized.pkl")

    # d = scn.load(local_scn_path="/home/jok120/scn221001/sidechainnet_debug12.pkl", trim_edges=False)
    seq_dict = {}
    # default_aligner = get_aligner()
    count = 0
    for pmin_id in tqdm(scn_minimized.get_pnids(), smoothing=0):
        p = d[pmin_id]
        make_new_alignment_dir(new_alignment_dir, p.id)
        roda_dirname = get_roda_dirname(roda_alignment_dir, p.id)

        if roda_dirname is None:
            # If we can't find a corresponding directory, this is likely a CASP target
            print(f"Unable to process {p.id}")
            continue

        # if count > 10:
        #     break

        # write_new_query_sequence_to_fasta_file(input_fasta_file, p)
        count += 1

        # Loop through all alignments for this protein
        for alignment_name in glob.glob(os.path.join(roda_dirname, 'a3m', "*.a3m")):
            output_filename = os.path.join(new_alignment_dir, p.id,
                                           os.path.basename(alignment_name))
            if os.path.isfile(output_filename):
                # If we've already processed this alignment, skip it
                print(f"Already processed {p.id}.")
                break
            query_seq = get_query_sequence(alignment_name)
            seq_dict[os.path.basename(os.path.normpath(roda_dirname))] = (p.seq,
                                                                          query_seq)

            is_identical = check_if_scnseq_is_identical_to_query_seq(query_seq, p)
            # If the query and SCN sequence are identical, we can just copy the file
            if is_identical == "True":
                # identical_file.write(f"{p.id} is identical to {alignment_name}.\n")
                shutil.copy2(alignment_name, output_filename)

                try:
                    source_hhr_file = glob.glob(os.path.join(roda_dirname, 'hhr', "*.hhr"))[0]
                except IndexError:
                    # There is no hhr file for this alignment
                    continue
                target_hhr_file = os.path.join(new_alignment_dir, p.id,
                                               os.path.basename(source_hhr_file))
                # Check if file exists; if it does not exist, copy it
                if os.path.isfile(source_hhr_file) and not os.path.isfile(target_hhr_file):
                    shutil.copy2(source_hhr_file, target_hhr_file)
                # alignment = align_and_select_cols(alignment_name, p)
                # write_new_alignment(alignment, output_filename, 'clustal')
                # with open(f"/scr/experiments/221101/fastas/{p.id}.fasta", "w") as f:
                #     f.write(f">{p.id}\n{p.seq}\n")

                continue
            # If the query is a subsequence of the SCN sequence, we trim the alignment
            # elif is_identical == "Subsequence":
            # subsequence_file.write(f"{p.id} is a subsequence of {alignment_name}.\n")
            # new_alignment = next(default_aligner.align(p.seq, query_seq))
            # subsequence_file.write(str(new_alignment) + "\n")
            # alignment = trim_alignment(alignment_name, p)
            # If the query and SCN sequence differ, we align & select the correct MSA cols
            # else:
            # new_alignment = next(default_aligner.align(p.seq, query_seq))
            # not_identifcal_file.write(
            # f"{p.id} is not identical to {alignment_name}.\n")
            # not_identifcal_file.write(str(new_alignment) + "\n")

            # if len(p.seq) < len(query_seq):
            # unreported_gap_file.write(
            # f"{p.id} is not identical to {alignment_name}.\n")
            # unreported_gap_file.write(str(new_alignment) + "\n")

            # alignment = align_and_select_cols(alignment_name, p)
            # write_new_alignment(alignment, output_filename, format='clustal')

    # identical_file.close()
    # not_identifcal_file.close()
    # subsequence_file.close()
    # input_fasta_file.close()

    # with open("seq_dict.pkl", "wb") as f:
    #     pickle.dump(seq_dict, f)


if __name__ == "__main__":
    roda_dir = "/scr/roda/pdb"
    new_dir = "/scr/scn_roda"
    main(roda_dir, new_dir)