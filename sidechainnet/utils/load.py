"""Implements SidechainNet loading functionality."""

import os
import pickle

import prody as pr
import requests
import sidechainnet as scn
import tqdm
from sidechainnet.create import format_sidechainnet_path
from sidechainnet.dataloaders.collate import prepare_dataloaders
from sidechainnet.dataloaders.SCNDataset import SCNDataset
from sidechainnet.dataloaders.SCNProtein import SCNProtein
from sidechainnet.utils.download import get_resolution_from_pdbid


def _get_local_sidechainnet_path(casp_version, thinning, scn_dir):
    """Return local path to SidechainNet file iff it exists, else returns None."""
    filepath = os.path.join(scn_dir, format_sidechainnet_path(casp_version, thinning))
    if os.path.isfile(filepath):
        return filepath
    else:
        return None


def _copyfileobj(fsrc, fdst, length=0, chunks=0.):
    """Copy data from file-like object fsrc to file-like object fdst.

    Modified from shutil.copyfileobj to include a progress bar with tqdm.
    """
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


def _download(url, file_name):
    """Download a file at a given URL to a specified local file_name with shutil."""
    # File length can only be approximated from the resulting GET, unfortunately
    r = requests.get(url, stream=True)
    if 'Content-Length' in r.headers:
        file_len = int(r.headers['Content-Length'])
    elif 'X-Original-Content-Length' in r.headers:
        file_len = int(r.headers['X-Original-Content-Length'])
    else:
        file_len = 0
    r.raw.decode_content = True
    with open(file_name, 'wb') as f:
        _copyfileobj(r.raw, f, chunks=(file_len / (64. * 1024)))
    r.close()

    return file_name


def _download_sidechainnet(casp_version, thinning, scn_dir):
    """Download the specified version of Sidechainnet."""
    # Prepare destination paths for downloading
    if format_sidechainnet_path(casp_version, thinning) not in SCN_URLS:
        raise FileNotFoundError(
            "The requested file is currently unavailable. Please check back later.")
    outfile_path = os.path.join(scn_dir, format_sidechainnet_path(casp_version, thinning))
    os.makedirs(os.path.dirname(outfile_path), exist_ok=True)
    print("Downloading from", SCN_URLS[format_sidechainnet_path(casp_version, thinning)])

    # Use a data-agnostic tool for downloading URL data from web to a specified local file
    _download(SCN_URLS[format_sidechainnet_path(casp_version, thinning)], outfile_path)
    print(f"Downloaded SidechainNet to {outfile_path}.")

    return outfile_path


def _load_dict(local_path):
    """Load a pickled dictionary."""
    with open(local_path, "rb") as f:
        d = pickle.load(f)
    print(f"SidechainNet was loaded from {local_path}.")
    return d


def load(casp_version=12,
         casp_thinning=30,
         scn_dir="./sidechainnet_data",
         force_download=False,
         with_pytorch=None,
         aggregate_model_input=True,
         collate_fn=None,
         batch_size=32,
         seq_as_onehot=None,
         dynamic_batching=True,
         num_workers=2,
         optimize_for_cpu_parallelism=False,
         train_eval_downsample=.2,
         filter_by_resolution=False,
         complete_structures_only=False,
         local_scn_path=None,
         scn_dataset=True,
         shuffle=True,
         trim_edges=False,
         **kwargs):
    #: Okay
    """Load and return the specified SidechainNet dataset in the specified manner.

    This function flexibly allows the user to load SidechainNet in a format that is most
    convenient to them. The user can specify which version and "thinning" of the dataset
    to load, and whether or not they would like the data prepared as a PyTorch DataLoader
    (with_pytorch='dataloaders') for easy access for model training with PyTorch. Several
    arguments are also available to allow the user to specify how the data should be
    loaded and batched when provided as DataLoaders (aggregate_model_input, collate_fn,
    batch_size, seq_as_one_hot, dynamic_batching, num_workers,
    optimize_for_cpu_parallelism, and train_eval_downsample.)

    By default, the data is returned as a SCNDataset object, to facilitate inspection.

    Args:
        casp_version (int, optional): CASP version to load (7-12). Defaults to 12.
        casp_thinning (int, optional): ProteinNet/SidechainNet "thinning" to load. A
            thinning represents the minimum sequence similarity each protein sequence must
            have to all other sequences in the same thinning. The 100 thinning contains
            all of the protein entries in SidechainNet, while the 30 thinning has a much
            smaller amount. Defaults to 30.
        scn_dir (str, optional): Path where SidechainNet data will be stored locally.
            Defaults to "./sidechainnet_data".
        force_download (bool, optional): If true, download SidechainNet data from the web
            even if it already exists locally. Defaults to False.
        with_pytorch (str, optional): If equal to 'dataloaders', returns a dictionary
            mapping dataset splits (e.g. 'train', 'test', 'valid-X') to PyTorch
            DataLoaders for data batching and model training. Defaults to None.
        aggregate_model_input (bool, optional): If True, the batches in the DataLoader
            contain a single entry for all of the SidechainNet data that is favored for
            use in a predictive model (sequences and PSSMs). This entry is a single
            Tensor. However, if False, when batching these entries are returned
            separately. See method description. Defaults to True.
        collate_fn (Callable, optional): A collating function. Defaults to None. See:
            https://pytorch.org/docs/stable/data.html#dataloader-collate-fn.
        batch_size (int, optional): Batch size to be used with PyTorch DataLoaders. Note
            that if dynamic_batching is True, then the size of the batch will not
            necessarily be equal to this number (though, on average, it will be close
            to this number). Only applicable when with_pytorch='dataloaders' is provided.
            Defaults to 32.
        seq_as_onehot (bool, optional): By default, the None value of this argument causes
            sequence data to be represented as one-hot vectors (L x 20) when batching and
            aggregate_model_input=True or to be represented as integer sequences (shape L,
            values 0 through 21 with 21 being a pad character). The user may override this
            option with seq_as_onehot=False only when aggregate_model_input=False.
        dynamic_batching (bool, optional): If True, uses a dynamic batch size when
            training that increases when the proteins within a batch have short sequences
            or decreases when the proteins within a batch have long sequences. Behind the
            scenes, this function bins the sequences in the training Dataset/DataLoader
            by their length. For every batch, it selects a bin at random (with a
            probability proportional to the number of proteins within that bin), and then
            selects N proteins within that batch, where:
                N = (batch_size * average_length_in_dataset)/max_length_in_bin.
            This means that, on average, each batch will have about the same number of
            amino acids. If False, uses a constant value (specified by batch_size) for
            batch size.
        num_workers (int, optional): Number of workers passed to DataLoaders. Defaults to
            2. See the description of workers in the PyTorch documentation:
            https://pytorch.org/docs/stable/data.html#single-and-multi-process-data-loading.
        optimize_for_cpu_parallelism (bool, optional): If True, ensure that the size of
            each batch is a multiple of the number of available CPU cores. Defaults to
            False.
        train_eval_downsample (float, optional): The fraction of the training set to
            include in the 'train-eval' DataLoader/Dataset that is returned. This is
            included so that, instead of evaluating the entire training set during each
            epoch of training (which can be expensive), we can first downsample the
            training set at the start of training, and use that downsampled dataset during
            the whole of model training. Defaults to .2.
        filter_by_resolution (float, bool, optional): If True, only use structures with a
            reported resolution < 3 Angstroms. Structures with no reported resolutions
            will also be excluded. If filter_by_resolution is a float, then only
            structures having a resolution value LESS than or equal this threshold will be
            included. For example, a value of 2.5 will exclude all structures with
            resolution greater than 2.5 Angstrom. Only the training set is filtered.
        complete_structures_only (bool, optional): If True, yield only structures from the
            training set that have no missing residues. Filter not applied to other data
            splits. Default False.
        local_scn_path (str, optional): The path for a locally saved SidechainNet file.
            This is especially useful for loading custom SidechainNet datasets.
        scn_dataset (bool, optional): If True, return a sidechainnet.SCNDataset object
            for conveniently accessing properties of the data. Default True.
            (See sidechainnet.SCNDataset) for more information.
        shuffle (bool, optional): Default True. If True, yields random batches from the
            dataloader instead of in-order (length ascending). Does not apply when a
            dataloader is not requested.
        trim_edges (bool, optional): If True, trim missing residues from the beginning
            and end of the protein sequence. Default False.

    Returns:
        A Python dictionary that maps data splits ('train', 'test', 'train-eval',
        'valid-X') to either more dictionaries containing protein data ('seq', 'ang',
        'crd', etc.) or to PyTorch DataLoaders that can be used for training. See below.
        
        Option 0 (SCNDataset, default):
            By default, scn.load returns a SCNDataset object. This object has the
            following properties:
                - indexing by integer or ProteinNet IDs yields SCNProtein objects.
                - is iterable.

        Option 1 (Python dictionary):
            If scn_dataset=False, the function returns a dictionary that is organized by
            training/validation/testing splits. For example, the following code loads CASP
            12 with the 30% thinning option:

                >>> import sidechainnet as scn
                >>> data = scn.load(12, 30)

            `data` is a Python dictionary with the following structure:

                data = {"train": {"seq": [seq1, seq2, ...],  # Sequences
                        "ang": [ang1, ang2, ...],  # Angles
                        "crd": [crd1, crd2, ...],  # Coordinates
                        "evo": [evo1, evo2, ...],  # PSSMs and Information Content
                        "ids": [id1, id2,   ...],  # Corresponding ProteinNet IDs
                        },
                        "valid-10": {...},
                            ...
                        "valid-90": {...},
                        "test":     {...},
                        "settings": {...},
                        "description" : "SidechainNet for CASP 12."
                        "date": "September 20, 2020"
                        }

        Option 2 (PyTorch DataLoaders):
            Alternatively, if the user provides `with_pytorch='dataloaders'`, `load` will
            return a dictionary mapping dataset "splits" (e.g. 'train', 'test', 'valid-X'
            where 'X' is one of the validation set splits defined by ProteinNet/
            SidechainNet).

            By default, the provided `DataLoader`s use a custom batching method that
            randomly generates batches of proteins of similar length for faster training.
            The probability of selecting small-length batches is decreased so that each
            protein in SidechainNet is included in a batch with equal probability. See
            `dynamic_batching` and  `collate_fn` arguments for more information on
            modifying this behavior. In the example below, `model_input` is a collated
            Tensor containing sequence and PSSM information.

                >>> dataloaders = scn.load(casp_version=12, with_pytorch="dataloaders")
                >>> dataloaders.keys()
                ['train', 'train_eval', 'valid-10', ..., 'valid-90', 'test']
                >>> for (protein_id, protein_seqs, model_input, true_angles,
                        true_coords) in dataloaders['train']:
                ....    predicted_angles = model(model_input)
                ....    predicted_coords = angles_to_coordinates(predicted_angles)
                ....    loss = compute_loss(predicted_angles, predicted_coords,
                                            true_angles, true_coords)
                ....    ...

            We have also made it possible to access the protein sequence and PSSM data
            directly when training by adding `aggregate_model_input=False` to `scn.load`.

                >>> dataloaders = scn.load(casp_version=12, with_pytorch="dataloaders",
                                        aggregate_model_input=False)
                >>> for (protein_id, sequence, pssm, true_angles,
                        true_coords) in dataloaders['train']:
                ....    prediction = model(sequence, pssm)
                ....    ...
    """
    if local_scn_path:
        local_path = local_scn_path
    else:
        local_path = _get_local_sidechainnet_path(casp_version, casp_thinning, scn_dir)
        if not local_path:
            print(f"SidechainNet{(casp_version, casp_thinning)} was not found in "
                  "{scn_dir}.")
    if not local_path or force_download:
        # Download SidechainNet if it does not exist locally, or if requested
        local_path = _download_sidechainnet(casp_version, casp_thinning, scn_dir)

    try:
        scn_dict = _load_dict(local_path)
    except pickle.UnpicklingError:
        print("Redownloading due to Pickle file error.")
        local_path = _download_sidechainnet(casp_version, casp_thinning, scn_dir)
        scn_dict = _load_dict(local_path)

    scn_dict = filter_dictionary_by_resolution(scn_dict, threshold=filter_by_resolution)

    # By default, the load function returns a dictionary
    if not with_pytorch and not scn_dataset:
        return scn_dict
    elif not with_pytorch and scn_dataset:
        return SCNDataset(scn_dict,
                          complete_structures_only=complete_structures_only,
                          trim_edges=trim_edges)
    if with_pytorch == "dataloaders":
        return prepare_dataloaders(
            scn_dict,
            aggregate_model_input=aggregate_model_input,
            collate_fn=collate_fn,
            batch_size=batch_size,
            num_workers=num_workers,
            seq_as_onehot=seq_as_onehot,
            dynamic_batching=dynamic_batching,
            optimize_for_cpu_parallelism=optimize_for_cpu_parallelism,
            train_eval_downsample=train_eval_downsample,
            shuffle=shuffle,
            complete_structures_only=complete_structures_only)

    return


def filter_dictionary_by_resolution(raw_data, threshold=False):
    """Filter SidechainNet data by removing poor-resolution training entries.

    Args:
        raw_data (dict): SidechainNet dictionary.
        threshold (float, bool): Entries with resolution values greater than this value
            are discarded. Test set entries have no measured resolution and are not
            excluded. Default is 3 Angstroms. If False, nothing is filtered.

    Returns:
        Filtered dictionary.
    """
    if not threshold:
        return raw_data
    if isinstance(threshold, bool) and threshold is True:
        threshold = 3
    new_data = {
        "seq": [],
        "ang": [],
        "ids": [],
        "evo": [],
        "msk": [],
        "crd": [],
        "sec": [],
        "res": [],
        "ums": [],
        "mod": []
    }
    train = raw_data["train"]
    n_filtered_entries = 0
    total_entires = 0.
    for seq, ang, crd, msk, evo, _id, res, sec, ums, mod in zip(
            train['seq'], train['ang'], train['crd'], train['msk'], train['evo'],
            train['ids'], train['res'], train['sec'], train['ums'], train['mod']):
        total_entires += 1
        if not res or res > threshold:
            n_filtered_entries += 1
            continue
        else:
            new_data["seq"].append(seq)
            new_data["ang"].append(ang)
            new_data["ids"].append(_id)
            new_data["evo"].append(evo)
            new_data["msk"].append(msk)
            new_data["crd"].append(crd)
            new_data["sec"].append(sec)
            new_data["res"].append(res)
            new_data["ums"].append(ums)
            new_data["mod"].append(mod)
    if n_filtered_entries:
        print(f"{n_filtered_entries} ({n_filtered_entries/total_entires:.1%})"
              " training set entries were excluded based on resolution.")
    raw_data["train"] = new_data
    return raw_data


def filter_dictionary_by_missing_residues(raw_data):
    """Return new SidechainNet dictionary that omits training data with missing residues.

    Args:
        raw_data (dict): SidechainNet dictionary.

    Returns:
        Filtered dictionary.
    """
    new_data = {
        "seq": [],
        "ang": [],
        "ids": [],
        "evo": [],
        "msk": [],
        "crd": [],
        "sec": [],
        "res": [],
        "ums": [],
        "mod": []
    }
    train = raw_data["train"]
    n_filtered_entries = 0
    total_entires = 0.
    for seq, ang, crd, msk, evo, _id, res, sec, ums, mod in zip(
            train['seq'], train['ang'], train['crd'], train['msk'], train['evo'],
            train['ids'], train['res'], train['sec'], train['ums'], train['mod']):
        total_entires += 1
        if "-" in msk:
            n_filtered_entries += 1
            continue
        else:
            new_data["seq"].append(seq)
            new_data["ang"].append(ang)
            new_data["ids"].append(_id)
            new_data["evo"].append(evo)
            new_data["msk"].append(msk)
            new_data["crd"].append(crd)
            new_data["sec"].append(sec)
            new_data["res"].append(res)
            new_data["ums"].append(ums)
            new_data["mod"].append(mod)

    if n_filtered_entries:
        print(f"{n_filtered_entries} ({n_filtered_entries/total_entires:.1%})"
              " training set entries were excluded based on missing residues.")
    raw_data["train"] = new_data
    return raw_data


def load_pdb(filename, pdbid="", include_resolution=False, scnprotein=True):
    """Return a SCNProtein containing SidechainNet-relevant data for a given PDB file.

    Args:
        filename (str): Path to existing PDB file.
        pdbid (str): 4-letter string representing the PDB Identifier.
        include_resolution (bool, default=False): If True, query the PDB for the protein
            structure resolution based off of the given pdb_id.

    Returns:
        scndata (SCNProtein): A SCNProtein object containing information parsed from the
            specified PDB file. If scnprotein is set to false, the data is returned as a
            dictionary holding the parsed data attributes of the protein
            structure. Below is a description of the keys:

                The key 'seq' is a 1-letter amino acid sequence.
                The key 'coords' is a (L x NUM_COORDS_PER_RES) x 3 numpy array.
                The key 'angs' is a L x NUM_ANGLES numpy array.
                The key 'is_nonstd' is a L x 1 numpy array with binary values. 1
                    represents that the amino acid at that position was a non-standard
                    amino acid that has been modified by SidechainNet into its standard
                    form.
                The key 'unmodified_seq' refers to the original amino acid sequence
                    of the protein structure. Some non-standard amino acids are converted
                    into their standard form by SidechainNet before measurement. In this
                    case, the unmodified_seq variable will contain the original
                    (3-letter code) seq.
                The key 'resolution' is the resolution of the structure as listed on the
                    PDB.
    """
    # First, use Prody to parse the PDB file
    chain = pr.parsePDB(filename)
    # Next, use SidechainNet to make the relevant measurements given the Prody chain obj
    (dihedrals_np, coords_np, observed_sequence, unmodified_sequence,
     is_nonstd) = scn.utils.measure.get_seq_coords_and_angles(chain, replace_nonstd=True)
    scndata = {
        "coords": coords_np,
        "angs": dihedrals_np,
        "seq": observed_sequence,
        "unmodified_seq": unmodified_sequence,
        "is_nonstd": is_nonstd
    }
    # If requested, look up the resolution of the given PDB ID
    if include_resolution:
        assert pdbid, "You must provide a PDB ID to look up the resolution."
        scndata['resolution'] = get_resolution_from_pdbid(pdbid)

    if scnprotein:
        p = SCNProtein(coordinates=scndata['coords'].reshape(len(observed_sequence), -1,
                                                             3),
                       angles=scndata['angs'],
                       sequence=scndata['seq'],
                       unmodified_seq=scndata["unmodified_seq"],
                       is_modified=scndata["is_nonstd"],
                       mask='+' * len(observed_sequence),
                       id=pdbid)
        return p
    return scndata


_base_url = "http://bits.csb.pitt.edu/~jok120/sidechainnet_data/"
SCN_URLS = {
    # CASP 12
    "sidechainnet_casp12_30.pkl":  _base_url + "sidechainnet_casp12_30.pkl",
    "sidechainnet_casp12_50.pkl":  _base_url + "sidechainnet_casp12_50.pkl",
    "sidechainnet_casp12_70.pkl":  _base_url + "sidechainnet_casp12_70.pkl",
    "sidechainnet_casp12_90.pkl":  _base_url + "sidechainnet_casp12_90.pkl",
    "sidechainnet_casp12_95.pkl":  _base_url + "sidechainnet_casp12_95.pkl",
    "sidechainnet_casp12_100.pkl": _base_url + "sidechainnet_casp12_100.pkl",

    # CASP 11
    "sidechainnet_casp11_30.pkl":  _base_url + "sidechainnet_casp11_30.pkl",
    "sidechainnet_casp11_50.pkl":  _base_url + "sidechainnet_casp11_50.pkl",
    "sidechainnet_casp11_70.pkl":  _base_url + "sidechainnet_casp11_70.pkl",
    "sidechainnet_casp11_90.pkl":  _base_url + "sidechainnet_casp11_90.pkl",
    "sidechainnet_casp11_95.pkl":  _base_url + "sidechainnet_casp11_95.pkl",
    "sidechainnet_casp11_100.pkl": _base_url + "sidechainnet_casp11_100.pkl",

    # CASP 10
    "sidechainnet_casp10_30.pkl":  _base_url + "sidechainnet_casp10_30.pkl",
    "sidechainnet_casp10_50.pkl":  _base_url + "sidechainnet_casp10_50.pkl",
    "sidechainnet_casp10_70.pkl":  _base_url + "sidechainnet_casp10_70.pkl",
    "sidechainnet_casp10_90.pkl":  _base_url + "sidechainnet_casp10_90.pkl",
    "sidechainnet_casp10_95.pkl":  _base_url + "sidechainnet_casp10_95.pkl",
    "sidechainnet_casp10_100.pkl": _base_url + "sidechainnet_casp10_100.pkl",

    # CASP 9
    "sidechainnet_casp9_30.pkl":  _base_url + "sidechainnet_casp9_30.pkl",
    "sidechainnet_casp9_50.pkl":  _base_url + "sidechainnet_casp9_50.pkl",
    "sidechainnet_casp9_70.pkl":  _base_url + "sidechainnet_casp9_70.pkl",
    "sidechainnet_casp9_90.pkl":  _base_url + "sidechainnet_casp9_90.pkl",
    "sidechainnet_casp9_95.pkl":  _base_url + "sidechainnet_casp9_95.pkl",
    "sidechainnet_casp9_100.pkl": _base_url + "sidechainnet_casp9_100.pkl",

    # CASP 8
    "sidechainnet_casp8_30.pkl":  _base_url + "sidechainnet_casp8_30.pkl",
    "sidechainnet_casp8_50.pkl":  _base_url + "sidechainnet_casp8_50.pkl",
    "sidechainnet_casp8_70.pkl":  _base_url + "sidechainnet_casp8_70.pkl",
    "sidechainnet_casp8_90.pkl":  _base_url + "sidechainnet_casp8_90.pkl",
    "sidechainnet_casp8_95.pkl":  _base_url + "sidechainnet_casp8_95.pkl",
    "sidechainnet_casp8_100.pkl": _base_url + "sidechainnet_casp8_100.pkl",

    # CASP 7
    "sidechainnet_casp7_30.pkl":  _base_url + "sidechainnet_casp7_30.pkl",
    "sidechainnet_casp7_50.pkl":  _base_url + "sidechainnet_casp7_50.pkl",
    "sidechainnet_casp7_70.pkl":  _base_url + "sidechainnet_casp7_70.pkl",
    "sidechainnet_casp7_90.pkl":  _base_url + "sidechainnet_casp7_90.pkl",
    "sidechainnet_casp7_95.pkl":  _base_url + "sidechainnet_casp7_95.pkl",
    "sidechainnet_casp7_100.pkl": _base_url + "sidechainnet_casp7_100.pkl",

    # Other
    "sidechainnet_debug.pkl":     _base_url + "sidechainnet_debug.pkl",
}
