"""Implements SidechainNet loading functionality."""
import pickle
import os

import requests
import tqdm

from sidechainnet.create import format_sidechainnet_path
from sidechainnet.dataloaders.collate import prepare_dataloaders


def get_local_sidechainnet_path(casp_version, thinning, scn_dir):
    """Returns local path to SidechainNet file iff it exists, else returns None."""
    filepath = os.path.join(scn_dir, format_sidechainnet_path(casp_version, thinning))
    if os.path.isfile(filepath):
        return filepath
    else:
        return None


def copyfileobj(fsrc, fdst, length=0, chunks=0.):
    """copy data from file-like object fsrc to file-like object fdst.
    Modified from shutil.copyfileobj."""
    # Localize variable access to minimize overhead.
    if not length:
        length = 64 * 1024
    fsrc_read = fsrc.read
    fdst_write = fdst.write
    if chunks:
        pbar = tqdm.tqdm(total=int(chunks),
                         desc='Downloading file chunks (estimated)',
                         unit='chunk',
                         dynamic_ncols=True)
    while True:
        buf = fsrc_read(length)
        if not buf:
            break
        fdst_write(buf)
        if chunks:
            pbar.update()


def download(url, file_name):
    """Downloads a file at a given URL to a specified local file_name with shutil."""
    # File length can only be approximated from the resulting GET, unfortunately
    r = requests.get(url, stream=True)
    if 'Content-Length' in r.headers:
        l = int(r.headers['Content-Length'])
    elif 'X-Original-Content-Length' in r.headers:
        l = int(r.headers['X-Original-Content-Length'])
    else:
        l = 0
    r.raw.decode_content = True
    with open(file_name, 'wb') as f:
        copyfileobj(r.raw, f, chunks=(l / (64. * 1024)))
    r.close()

    return file_name


def download_sidechainnet(casp_version, thinning, scn_dir):
    """Downloads the specified version of Sidechainnet."""
    # Prepare destination paths for downloading
    if format_sidechainnet_path(casp_version, thinning) not in BOXURLS:
        raise FileNotFoundError(
            "The requested file is currently unavailable. Please check back later.")
    outfile_path = os.path.join(scn_dir, format_sidechainnet_path(casp_version, thinning))
    os.makedirs(os.path.dirname(outfile_path), exist_ok=True)
    print("Downloading from", BOXURLS[format_sidechainnet_path(casp_version, thinning)])

    # Use a data-agnostic tool for downloading URL data from Box to a specified local file
    download(BOXURLS[format_sidechainnet_path(casp_version, thinning)], outfile_path)
    print(f"Downloaded SidechainNet to {outfile_path}.")

    return outfile_path


def load_dict(local_path):
    """Loads a pickled dictionary."""
    with open(local_path, "rb") as f:
        d = pickle.load(f)
    print(f"SidechainNet was loaded from {local_path}.")
    return d


def load(casp_version=12,
         thinning=30,
         scn_dir="./sidechainnet",
         force_download=False,
         with_pytorch=None,
         aggregate_model_input=True,
         collate_fn=None,
         batch_size=32,
         return_masks=False,
         seq_as_onehot=None,
         use_dynamic_batch_size=True,
         num_workers=2,
         optimize_for_cpu_parallelism=False,
         train_eval_downsample=.2):
    """Loads SidechainNet as a Python dictionary.
    
    Args:
        force_download: Downloads SidechainNet data to regardless of whether or not it
            has already been downloaded. 
        casp_version: An integer between 7 and 12, representing which CASP contest (and 
            therefore which ProteinNet version) to load SidechainNet from.
        thinning: An integer (30, 50, 70, 90, 95, 100) representing the training set
            thinning to load. 100 means that 100% of the proteins will be loaded, while
            30 means that the precomputed 30% thinning of the data will be loaded.
        scn_dir: A string representing a local path to store the SidechainNet data files.
            By default, the data will be stored in the current directory, under a sub-
            directory title 'sidechainnet'.
        with_pytorch: Optional string argument specifying whether or not to load the
            SidechainNet data as PyTorch "dataloaders". This is the most convenient way
            to access the data for machine learning methods.
        aggregate_model_input: A boolean that, if True, yields batches of (protein_id,
            model_input, true_angles, true_coordinates) when iterating over the returned
            PyTorch DataLoader. If False, this expands the model_input variable into 
            its components (sequence, mask pssm).
    
    Returns:
        By default, this method returns a Python dictionary that contains SidechainNet
        organized by data splits (train, test, valid-X).    
    """
    local_path = get_local_sidechainnet_path(casp_version, thinning, scn_dir)
    if not local_path:
        print(f"SidechainNet{(casp_version, thinning)} was not found in {scn_dir}.")
    if not local_path or force_download:
        # Download SidechainNet if it does not exist locally, or if requested
        local_path = download_sidechainnet(casp_version, thinning, scn_dir)

    scn_dict = load_dict(local_path)

    # By default, the load function returns a dictionary
    if not with_pytorch:
        return scn_dict

    if with_pytorch == "dataloaders":
        return prepare_dataloaders(
            scn_dict,
            aggregate_model_input=aggregate_model_input,
            collate_fn=collate_fn,
            batch_size=batch_size,
            num_workers=num_workers,
            return_masks=return_masks,
            seq_as_onehot=seq_as_onehot,
            use_dynamic_batch_size=use_dynamic_batch_size,
            optimize_for_cpu_parallelism=optimize_for_cpu_parallelism,
            train_eval_downsample=train_eval_downsample)

    return


# TODO: Finish uploading files to Box for distribution
BOXURLS = {
    # CASP 12
    "sidechainnet_casp12_30.pkl":
        "https://pitt.box.com/shared/static/sjnecyqfm8hkqf0nk5yz1cf3ct4dygyf.pkl",
    "sidechainnet_casp12_50.pkl":
        "https://pitt.box.com/shared/static/h2sdnmxgobh9duamfzqvinjbx0tswn1i.pkl",
    "sidechainnet_casp12_70.pkl":
        "https://pitt.box.com/shared/static/7ikrp46bej1wylieqjpw76ceabcnc7r4.pkl",
    "sidechainnet_casp12_90.pkl":
        "https://pitt.box.com/shared/static/yn5ucsxwo9bp01yp23jltseg4j8sb3f8.pkl",
    "sidechainnet_casp12_95.pkl":
        "https://pitt.box.com/shared/static/k3fr29s5dyckj6f535silvqab6ikai43.pkl",
    "sidechainnet_casp12_100.pkl":
        "https://pitt.box.com/shared/static/52ggdgl60optmp57b5mcwocv4cabwmbv.pkl",

    # CASP 11
    "sidechainnet_casp11_30.pkl":
        "",
    "sidechainnet_casp11_50.pkl":
        "",
    "sidechainnet_casp11_70.pkl":
        "",
    "sidechainnet_casp11_90.pkl":
        "",
    "sidechainnet_casp11_95.pkl":
        "",
    "sidechainnet_casp11_100.pkl":
        "",

    # CASP 10
    "sidechainnet_casp10_30.pkl":
        "",
    "sidechainnet_casp10_50.pkl":
        "",
    "sidechainnet_casp10_70.pkl":
        "",
    "sidechainnet_casp10_90.pkl":
        "",
    "sidechainnet_casp10_95.pkl":
        "",
    "sidechainnet_casp10_100.pkl":
        "",

    # CASP 9
    "sidechainnet_casp9_30.pkl":
        "",
    "sidechainnet_casp9_50.pkl":
        "",
    "sidechainnet_casp9_70.pkl":
        "",
    "sidechainnet_casp9_90.pkl":
        "",
    "sidechainnet_casp9_95.pkl":
        "",
    "sidechainnet_casp9_100.pkl":
        "",

    # CASP 8
    "sidechainnet_casp8_30.pkl":
        "",
    "sidechainnet_casp8_50.pkl":
        "",
    "sidechainnet_casp8_70.pkl":
        "",
    "sidechainnet_casp8_90.pkl":
        "",
    "sidechainnet_casp8_95.pkl":
        "",
    "sidechainnet_casp8_100.pkl":
        "",

    # CASP 7
    "sidechainnet_casp7_30.pkl":
        "",
    "sidechainnet_casp7_50.pkl":
        "",
    "sidechainnet_casp7_70.pkl":
        "",
    "sidechainnet_casp7_90.pkl":
        "",
    "sidechainnet_casp7_95.pkl":
        "",
    "sidechainnet_casp7_100.pkl":
        ""
}
