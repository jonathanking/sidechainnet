"""Implements SidechainNet loading functionality."""

import pickle
import os
from sidechainnet.dataloaders.SCNDataset import SCNDataset

import requests
import tqdm

import sidechainnet as scn
from sidechainnet.create import format_sidechainnet_path
from sidechainnet.dataloaders.collate import prepare_dataloaders


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
    if format_sidechainnet_path(casp_version, thinning) not in BOXURLS:
        raise FileNotFoundError(
            "The requested file is currently unavailable. Please check back later.")
    outfile_path = os.path.join(scn_dir, format_sidechainnet_path(casp_version, thinning))
    os.makedirs(os.path.dirname(outfile_path), exist_ok=True)
    print("Downloading from", BOXURLS[format_sidechainnet_path(casp_version, thinning)])

    # Use a data-agnostic tool for downloading URL data from Box to a specified local file
    _download(BOXURLS[format_sidechainnet_path(casp_version, thinning)], outfile_path)
    print(f"Downloaded SidechainNet to {outfile_path}.")

    return outfile_path


def _load_dict(local_path):
    """Load a pickled dictionary."""
    with open(local_path, "rb") as f:
        d = pickle.load(f)
    print(f"SidechainNet was loaded from {local_path}.")
    return d


def load(casp_version=12,
         thinning=30,
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
         scn_dataset=False):
    #: Okay
    """Load and return the specified SidechainNet dataset as a dictionary or DataLoaders.

    This function flexibly allows the user to load SidechainNet in a format that is most
    convenient to them. The user can specify which version and "thinning" of the dataset
    to load, and whether or not they would like the data prepared as a PyTorch DataLoader
    (with_pytorch='dataloaders') for easy access for model training with PyTorch. Several
    arguments are also available to allow the user to specify how the data should be
    loaded and batched when provided as DataLoaders (aggregate_model_input, collate_fn,
    batch_size, seq_as_one_hot, dynamic_batching, num_workers,
    optimize_for_cpu_parallelism, and train_eval_downsample.)

    Args:
        casp_version (int, optional): CASP version to load (7-12). Defaults to 12.
        thinning (int, optional): ProteinNet/SidechainNet "thinning" to load. A thinning
            represents the minimum sequence similarity each protein sequence must have to
            all other sequences in the same thinning. The 100 thinning contains all of the
            protein entries in SidechainNet, while the 30 thinning has a much smaller
            amount. Defaults to 30.
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
            reported resolution < 3 Angstroms. Structures wit no reported resolutions will
            also be excluded. If filter_by_resolution is a float, then only structures
            having a resolution value LESS than or equal this threshold will be included.
            For example, a value of 2.5 will exclude all structures with resolution
            greater than 2.5 Angstrom. Only the training set is filtered.
        complete_structures_only (bool, optional): If True, yield only structures from the
            training set that have no missing residues. Filter not applied to other data
            splits. Default False.
        local_scn_path (str, optional): The path for a locally saved SidechainNet file.
            This is especially useful for loading custom SidechainNet datasets.
        scn_dataset (bool, optional): If True, return a sidechainnet.SCNDataset object
            for conveniently accessing properties of the data.
            (See sidechainnet.SCNDataset) for more information.

    Returns:
        A Python dictionary that maps data splits ('train', 'test', 'train-eval',
        'valid-X') to either more dictionaries containing protein data ('seq', 'ang',
        'crd', etc.) or to PyTorch DataLoaders that can be used for training. See below.

        Option 1 (Python dictionary):
            By default, the function returns a dictionary that is organized by training/
            validation/testing splits. For example, the following code loads CASP 12 with
            the 30% thinning option:

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
        local_path = _get_local_sidechainnet_path(casp_version, thinning, scn_dir)
        if not local_path:
            print(f"SidechainNet{(casp_version, thinning)} was not found in {scn_dir}.")
    if not local_path or force_download:
        # Download SidechainNet if it does not exist locally, or if requested
        local_path = _download_sidechainnet(casp_version, thinning, scn_dir)

    scn_dict = _load_dict(local_path)

    # Patch for removing 1GJJ_1_A, see Issue #38
    scn_dict = scn.utils.manual_adjustment._repair_1GJJ_1_A(scn_dict)

    scn_dict = filter_dictionary_by_resolution(scn_dict, threshold=filter_by_resolution)
    if complete_structures_only:
        scn_dict = filter_dictionary_by_missing_residues(scn_dict)

    # By default, the load function returns a dictionary
    if not with_pytorch and not scn_dataset:
        return scn_dict
    elif not with_pytorch and scn_dataset:
        return SCNDataset(scn_dict)

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
            train_eval_downsample=train_eval_downsample)

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


BOXURLS = {
    # CASP 12
    "sidechainnet_casp12_30.pkl":
        "https://pitt.box.com/shared/static/hbatd2a750tx8e27yizwinc3hsceeeui.pkl",
    "sidechainnet_casp12_50.pkl":
        "https://pitt.box.com/shared/static/7cng5zdi2s4doruh1m512d281w2cmk0z.pkl",
    "sidechainnet_casp12_70.pkl":
        "https://pitt.box.com/shared/static/xfaktrj8ole0eqktxi5fa4qp9efum8f2.pkl",
    "sidechainnet_casp12_90.pkl":
        "https://pitt.box.com/shared/static/nh7vybjjm224m1nezrgmnywxsa4st2uk.pkl",
    "sidechainnet_casp12_95.pkl":
        "https://pitt.box.com/shared/static/wcz1kex8idnpy8zx7a59r3h6e216tlq1.pkl",
    "sidechainnet_casp12_100.pkl":
        "https://pitt.box.com/shared/static/ey5xh6l4p8iwzrxtxwpxt7oeg70eayl4.pkl",

    # CASP 11
    "sidechainnet_casp11_30.pkl":
        "https://pitt.box.com/shared/static/fzil4bgxt4fqpp416xw0e3y0ew4c7yct.pkl",
    "sidechainnet_casp11_50.pkl":
        "https://pitt.box.com/shared/static/rux3p18k523y8zbo40u1l856826buvui.pkl",
    "sidechainnet_casp11_70.pkl":
        "https://pitt.box.com/shared/static/tl51ym0hzjdvq4qs5f5shsj0sl9mkvd0.pkl",
    "sidechainnet_casp11_90.pkl":
        "https://pitt.box.com/shared/static/iheqs3vqszoxsdq46nkzf5kylt8ecjbx.pkl",
    "sidechainnet_casp11_95.pkl":
        "https://pitt.box.com/shared/static/gbme2a5yifpugtmthwu2989xxyg5b8i6.pkl",
    "sidechainnet_casp11_100.pkl":
        "https://pitt.box.com/shared/static/3cfx02k2yw4ux2mrbvwrrj91zsftcpbj.pkl",

    # CASP 10
    "sidechainnet_casp10_30.pkl":
        "https://pitt.box.com/shared/static/fe0hpjrldi2y1g374mgdzfpdipajd6s4.pkl",
    "sidechainnet_casp10_50.pkl":
        "https://pitt.box.com/shared/static/tsnt6s07txas0h37cpzepck580yme9vv.pkl",
    "sidechainnet_casp10_70.pkl":
        "https://pitt.box.com/shared/static/awmzr4jj68p61ab031smixryt69p8ykm.pkl",
    "sidechainnet_casp10_90.pkl":
        "https://pitt.box.com/shared/static/it6zcugy997c1550kima3m3fu8kamnh8.pkl",
    "sidechainnet_casp10_95.pkl":
        "https://pitt.box.com/shared/static/q6ld9h276kobhmmtvdq581qnm61oevup.pkl",
    "sidechainnet_casp10_100.pkl":
        "https://pitt.box.com/shared/static/fpixgzh9n86xyzpwtlc74lle4fd3p5es.pkl",

    # CASP 9
    "sidechainnet_casp9_30.pkl":
        "https://pitt.box.com/shared/static/j1h3181d2mibqvc7jrqm17dprzj6pxmc.pkl",
    "sidechainnet_casp9_50.pkl":
        "https://pitt.box.com/shared/static/l363lu9ztpdmcybthtytwnrvvkib2228.pkl",
    "sidechainnet_casp9_70.pkl":
        "https://pitt.box.com/shared/static/4uh1yggpdhm0aoeisomnyfuac4j20qzc.pkl",
    "sidechainnet_casp9_90.pkl":
        "https://pitt.box.com/shared/static/scv7l6qfr2j93pn4cu40ouhmxbns6k7x.pkl",
    "sidechainnet_casp9_95.pkl":
        "https://pitt.box.com/shared/static/tqpugpr7wamvmkyrtd8tqnzft6u53zha.pkl",
    "sidechainnet_casp9_100.pkl":
        "https://pitt.box.com/shared/static/jjtubu2lxwlv1aw8tfc7u27vcf2yz39v.pkl",

    # CASP 8
    "sidechainnet_casp8_30.pkl":
        "https://pitt.box.com/shared/static/1hx2n3y2gn3flnlsw2wb1e4l4nlru5mz.pkl",
    "sidechainnet_casp8_50.pkl":
        "https://pitt.box.com/shared/static/4u8tuqkm5pv34hm139uw9dqc4ieebsue.pkl",
    "sidechainnet_casp8_70.pkl":
        "https://pitt.box.com/shared/static/vj58yaeph55zjb04jezmqams66mn4bil.pkl",
    "sidechainnet_casp8_90.pkl":
        "https://pitt.box.com/shared/static/1ry2j47lde7zk5fxzvuffv05k1gq29oh.pkl",
    "sidechainnet_casp8_95.pkl":
        "https://pitt.box.com/shared/static/9uaw2tv61xyfd8gtw9n8e3hfcken4t4x.pkl",
    "sidechainnet_casp8_100.pkl":
        "https://pitt.box.com/shared/static/crk59vz6dw9cbbvne10owa450zgv1j79.pkl",

    # CASP 7
    "sidechainnet_casp7_30.pkl":
        "https://pitt.box.com/shared/static/hjblmbwei2dkwhfjatttdmamznt1k9ef.pkl",
    "sidechainnet_casp7_50.pkl":
        "https://pitt.box.com/shared/static/4pw56huei1123a5rd6g460886kg0pex7.pkl",
    "sidechainnet_casp7_70.pkl":
        "https://pitt.box.com/shared/static/afyow2ki9mwuoago0bzlsp5ame8dq12g.pkl",
    "sidechainnet_casp7_90.pkl":
        "https://pitt.box.com/shared/static/phsbdw8bj1oiv61d6hps0j62324820f3.pkl",
    "sidechainnet_casp7_95.pkl":
        "https://pitt.box.com/shared/static/2lgbtdw6c5df0qpe7dtnlaawowy9ic5r.pkl",
    "sidechainnet_casp7_100.pkl":
        "https://pitt.box.com/shared/static/6qipxz2z2n12a06vln5ucmzu4dcyw5ee.pkl",

    # Other
    "sidechainnet_debug.pkl":
        "https://pitt.box.com/shared/static/tevlb6nuii6kk520vi4x0u7li0eoxuep.pkl"
}
