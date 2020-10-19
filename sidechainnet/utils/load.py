"""Implements SidechainNet loading functionality."""
import pickle
import os

import requests
import tqdm

from sidechainnet.create import format_sidechainnet_path


def get_local_sidechainnet_path(casp_version, thinning, scn_dir):
    """Returns local path to SidechainNet file iff it exists, else returns None."""
    if os.path.isdir(scn_dir):
        return os.path.join(scn_dir, format_sidechainnet_path(casp_version, thinning))
    else:
        return None


def copyfileobj(fsrc, fdst, length=0, chunks=100.):
    """copy data from file-like object fsrc to file-like object fdst.
    Modified from shutil.copyfileobj."""
    # Localize variable access to minimize overhead.
    if not length:
        length = 64 * 1024
    fsrc_read = fsrc.read
    fdst_write = fdst.write
    pbar = tqdm.tqdm(total=chunks,
                     desc='Downloading file chunks (over-estimated)',
                     unit='chunk',
                     dynamic_ncols=True)
    while True:
        buf = fsrc_read(length)
        if not buf:
            break
        fdst_write(buf)
        pbar.update()


def download(url, file_name):
    """Downloads a file at a given URL to a specified local file_name with shutil."""
    # File length can only be approximated from the resulting GET, unfortunately
    r = requests.get(url, stream=True)
    l = int(r.headers['X-Original-Content-Length'])
    r.raw.decode_content = True
    with open(file_name, 'wb') as f:
        copyfileobj(r.raw, f, chunks=(l / (64. * 1024)))
    r.close()

    return file_name


def download_sidechainnet(casp_version, thinning, scn_dir):
    """Downloads the specified version of Sidechainnet."""
    # Prepare destination paths for downloading
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
        return pickle.load(f)


def load(casp_version=12, thinning=30, scn_dir="./sidechainnet"):
    """Loads SidechainNet as a Python dictionary.
    
    Args:
        casp_version: An integer between 7 and 12, representing which CASP contest (and 
            therefore which ProteinNet version) to load SidechainNet from.
        thinning: An integer (30, 50, 70, 90, 95, 100) representing the training set
            thinning to load. 100 means that 100% of the proteins will be loaded, while
            30 means that the precomputed 30% thinning of the data will be loaded.
        scn_dir: A string representing a local path to store the SidechainNet data files.
            By default, the data will be stored in the current directory, under a sub-
            directory title 'sidechainnet'.
        
    
    Returns:
        By default, this method returns a Python dictionary that contains SidechainNet
        organized by data splits (train, test, valid-X).    
    """
    local_path = get_local_sidechainnet_path(casp_version, thinning, scn_dir)
    print(local_path)
    if not local_path:
        print(f"SidechainNet was not found in {scn_dir}.")
        # Download SidechainNet if it does not exist locally
        local_path = download_sidechainnet(casp_version, thinning, scn_dir)

    return load_dict(local_path)


# TODO: Finish uploading files to Box for distribution
BOXURLS = {
    "sidechainnet_casp12_50.pkl":
        "https://pitt.box.com/shared/static/2ux5agaejvvvtzjdvl6mts5uk89q77v9.pkl",
    "sidechainnet_casp12_30.pkl":
        "https://pitt.box.com/shared/static/2ux5agaejvvvtzjdvl6mts5uk89q77v9.pkl"
}
