"""This script runs AlphaFold2 on the set of proteins in a SidechainNet dataset."""

import os
import shlex
import sys
import subprocess
import sidechainnet as scn

# Variables used to run AlphaFold2
MMCIF_FILES = "/scr/alphafold_data/pdb_mmcif/mmcif_files/"
DATA_ROOT = "/scr/alphafold_data"
JACKHMMER = "/home/jok120/anaconda3/envs/openfold_venv/bin/jackhmmer"
HHBLITS = "/home/jok120/anaconda3/envs/openfold_venv/bin/hhblits"
HHSEARCH = "/home/jok120/anaconda3/envs/openfold_venv/bin/hhsearch"
KALIGN = "/home/jok120/anaconda3/envs/openfold_venv/bin/kalign"
ALIGN_DIR = "/scr/scn_roda"

META_FASTA_FILE = "/home/jok120/10#META2.fa"


def main():
    # Parse arguments
    if len(sys.argv) != 5:
        print(sys.argv)
        print("Usage: python run_af2_on_scn.py <scn_path> <scn_min_path> <fasta_out_dir> <pred_out_dir>")
        sys.exit(1)
    scn_path = sys.argv[1]
    scn_min_path = sys.argv[2]
    fasta_out_dir = sys.argv[3]
    pred_out_dir = sys.argv[4]
    os.makedirs(fasta_out_dir, exist_ok=True)
    os.makedirs(pred_out_dir, exist_ok=True)

    # Load SidechainNet
    scn_dataset = scn.load(local_scn_path=scn_path)

    # Load the minimized SidechainNet
    scn_minimized = scn.load(local_scn_path=scn_min_path, sort_by_length='ascending')

    # Create the required fasta files
    # scn_dataset.to_fastas(fasta_out_dir, ids=scn_minimized.get_pnids())
    scn_dataset.to_fasta(META_FASTA_FILE, ids=scn_minimized.get_pnids())

    # Start a new AlphaFold2 run on the command line
    command = create_af2_cmd(fasta_out_dir, pred_out_dir, META_FASTA_FILE)

    print("Starting AlphaFold2 run...")
    print(command)
    exit(1)

    # Run the command using subprocess.Popen
    with open("af2.log", "wb") as f:
        process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE)
        for c in iter(lambda: process.stdout.read(1), b""):
            sys.stdout.buffer.write(c)
            f.buffer.write(c)

    print("AlphaFold2 run complete.")


def create_af2_cmd(fasta_out_dir, pred_out_dir, meta_fasta_file=None, config_preset="model_1_ptm"):
    """Create the command to run AlphaFold2 via the OpenFold package."""
    meta_fasta_text = f"--use_meta_fasta_file={meta_fasta_file}" if meta_fasta_file is not None else ""
    cmd = f"""/home/jok120/anaconda3/envs/openfold_venv/bin/python \
        /scr/openfold/run_pretrained_openfold.py \
        {fasta_out_dir} \
        {MMCIF_FILES} \
        --output_dir {pred_out_dir} \
        --config_preset {config_preset} \
        --use_precomputed_alignments {ALIGN_DIR} \
        --uniref90_database_path {DATA_ROOT}/uniref90/uniref90.fasta \
        --mgnify_database_path {DATA_ROOT}/mgnify/mgy_clusters_2018_12.fa \
        --pdb70_database_path {DATA_ROOT}/pdb70/pdb70 \
        --uniclust30_database_path {DATA_ROOT}/uniclust30/uniclust30_2018_08/uniclust30_2018_08 \
        --bfd_database_path {DATA_ROOT}/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
        --model_device "cuda:0" \
        --jackhmmer_binary_path {JACKHMMER} \
        --hhblits_binary_path {HHBLITS} \
        --hhsearch_binary_path {HHSEARCH} \
        --kalign_binary_path {KALIGN} \
        --save_outputs \
        {meta_fasta_text}
    """
    return cmd


if __name__ == "__main__":
    sys.argv = [
        "run_af2_on_scn.py",
        "/home/jok120/scn221001/sidechainnet_casp12_100.pkl",
        "/home/jok120/scnmin221013/scn_minimized.pkl",
        "/scr/experiments/221114/fastas",
        "/scr/experiments/221114/out3"]
    main()
