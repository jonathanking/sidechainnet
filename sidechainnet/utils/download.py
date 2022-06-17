"""Functions for downloading protein structure data from RCSB PDB using the ProDy pkg."""

import multiprocessing
import os
import pickle
import pkg_resources
from glob import glob
import re
import requests
import sys
import zipfile

import prody as pr
import tqdm

import sidechainnet.utils.errors as errors
from sidechainnet.utils.measure import get_seq_coords_and_angles, no_nans_infs_allzeros
from sidechainnet.utils.parse import get_chain_from_astral_id, parse_astral_summary_file, parse_dssp_file

MAX_SEQ_LEN = 10_000  # An arbitrarily large upper-bound on sequence lengths

VALID_SPLITS_INTS = [10, 20, 30, 40, 50, 70, 90]
VALID_SPLITS = [f'valid-{s}' for s in VALID_SPLITS_INTS]
DATA_SPLITS = ['train', 'test'] + VALID_SPLITS

PN_VALID_SPLITS_INTS = [10, 20, 30, 40, 50, 70, 90]
PN_VALID_SPLITS = [f'valid-{s}' for s in PN_VALID_SPLITS_INTS]
PN_DATA_SPLITS = ['train', 'test'] + PN_VALID_SPLITS

D_AMINO_ACID_CODES = [
    "DAL", "DSN", "DTH", "DCY", "DVA", "DLE", "DIL", "MED", "DPR", "DPN", "DTY", "DTR",
    "DSP", "DGL", "DSG", "DGN", "DHI", "DLY", "DAR"
]
ASTRAL_ID_MAPPING = None
PROTEIN_DSSP_DATA = None


def _reinit_global_valid_splits(new_splits):
    """Reinitialize global validation split variables when customizing dataset splits."""
    global VALID_SPLITS
    global VALID_SPLITS_INTS
    global DATA_SPLITS
    print(f"Re-initializing validation set splits ({new_splits}).")
    VALID_SPLITS_INTS = new_splits
    VALID_SPLITS = [f'valid-{s}' for s in VALID_SPLITS_INTS]
    DATA_SPLITS = ['train', 'test'] + VALID_SPLITS


def _init_dssp_data():
    """Initialize global variables for secondary structure and ASTRAL database info."""
    global PROTEIN_DSSP_DATA
    global ASTRAL_ID_MAPPING
    PROTEIN_DSSP_DATA = parse_dssp_file(
        pkg_resources.resource_filename("sidechainnet",
                                        "resources/full_protein_dssp_annotations.json"))
    PROTEIN_DSSP_DATA.update(
        parse_dssp_file(
            pkg_resources.resource_filename(
                "sidechainnet", "resources/single_domain_dssp_annotations.json")))
    with open(
            pkg_resources.resource_filename("sidechainnet", "resources/astral_data.txt"),
            "r") as astral_file:
        ASTRAL_ID_MAPPING = parse_astral_summary_file(astral_file.read().splitlines())


def download_sidechain_data(pnids,
                            sidechainnet_out_dir,
                            casp_version,
                            thinning,
                            limit,
                            proteinnet_in,
                            regenerate_scdata=False,
                            output_name=None):
    """Download the sidechain data for the corresponding ProteinNet IDs.

    Args:
        pnids: List of ProteinNet IDs to download sidechain data for
        sidechainnet_out_dir: Path to directory for saving sidechain data
        casp_version: A string that describes the CASP version i.e. 'casp7'
        thinning: Which thinning of ProteinNet to extract (30, 50, 90, etc.)
        limit: An integer describing maximum number of proteins to process
        proteinnet_in: A string representing the path to processed proteinnet.
        regenerate_scdata: Boolean, if True then recreate the sidechain-only data even if
            it already exists.
        output_name: A string describing the filename. Defaults to
            "sidechain-only_{casp_version}_{thinning}.pkl".

    Returns:
        sc_data: Python dictionary `{pnid: {...}, ...}`
    """
    from sidechainnet.utils.organize import load_data, save_data

    # Initialize directories.
    global PROTEINNET_IN_DIR
    PROTEINNET_IN_DIR = proteinnet_in
    if output_name is None:
        output_name = f"sidechain-only_{casp_version}_{thinning}.pkl"
    output_path = os.path.join(sidechainnet_out_dir, output_name)
    if not os.path.exists(sidechainnet_out_dir):
        os.makedirs(sidechainnet_out_dir)

    # Simply load sidechain data if it has already been processed.
    if os.path.exists(output_path) and not regenerate_scdata:
        print(f"Sidechain information already preprocessed ({output_path}).")
        return load_data(output_path), output_path

    print("Downloading SidechainNet specific data from RSCB PDB.")

    # Clean up any error logs that have been left-over from previous runs.
    if os.path.exists("errors"):
        for file in glob('errors/*.txt'):
            os.remove(file)
    else:
        os.makedirs("errors")

    # Try loading pre-parsed data from CASP12/100 to speed up dataset generation.
    already_downloaded_data = "sidechain-only_12_100.pkl"
    already_downloaded_data = os.path.join(sidechainnet_out_dir, already_downloaded_data)
    new_pnids = [p for p in pnids]
    already_parsed_ids = []
    if os.path.exists(already_downloaded_data):
        with open(already_downloaded_data, "rb") as f:
            existing_data = pickle.load(f)
        already_parsed_ids = [p for p in pnids if p in existing_data]
        new_pnids = [p for p in pnids if p not in existing_data]
        print(f"Will download {len(pnids)-len(new_pnids)} fewer pnids that were already"
              " processed.")

    # Download the sidechain data as a dictionary and report errors.
    sc_data, pnids_errors = get_sidechain_data(new_pnids, limit)
    for p in already_parsed_ids:
        sc_data[p] = existing_data[p]
    save_data(sc_data, output_path)
    errors.report_errors(pnids_errors, total_pnids=len(pnids[:limit]))

    # Clean up working directory
    for file in glob('*.cif'):
        os.remove(file)

    return sc_data, output_path


def get_sidechain_data(pnids, limit):
    """Acquires sidechain data for specified ProteinNet IDs.

    Args:
        pnids: List of ProteinNet IDs to download data for.
        limit: Number of IDs to process (use small value for debugging).

    Returns:
        Dictionary mapping pnids to sidechain data. For example:

        {"1a9u_A_2": {"seq": "MRYK...",
                      "ang": np.ndarray(...),
                        ...
                      }}
        Also returns a list of tuples of (pnid, error_code) for those pnids
        that failed to download.
    """

    # Downloading ProteinNet IDs is not thread safe since multiple pnids have the
    # same PDB ID and may attempt to download the file simultaneously.

    def get_parallel_sequential(pnid_list):
        pnids_ok_parallel = []
        pnids_sequential = []
        existing_pdbids = set()
        for p in pnid_list:
            if determine_pnid_type(p) == "test":
                pnids_ok_parallel.append(p)
                continue
            pdbid = get_pdbid_from_pnid(p)
            if pdbid not in existing_pdbids:
                existing_pdbids.add(pdbid)
                pnids_ok_parallel.append(p)
            else:
                pnids_sequential.append(p)
        return pnids_ok_parallel, pnids_sequential

    # First, we take a set of pnids with unique PDB IDs. These can be simultaneously
    # downloaded.
    results = []
    remaining_pnids = [_ for _ in pnids]
    pnids_ok_parallel, remaining_pnids = get_parallel_sequential(remaining_pnids)
    while len(remaining_pnids) > multiprocessing.cpu_count():
        print(f"{len(pnids_ok_parallel)} IDs OK for parallel downloading.")
        with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
            results.extend(
                list(
                    tqdm.tqdm(p.imap(process_id, pnids_ok_parallel[:limit]),
                              total=len(pnids_ok_parallel[:limit]),
                              dynamic_ncols=True,
                              smoothing=0)))
        pnids_ok_parallel, remaining_pnids = get_parallel_sequential(remaining_pnids)

    # Next, we can download the remaining pnids in sequential order safely.
    print("Downloading remaining", len(remaining_pnids), " sequentially.")

    pbar = tqdm.tqdm(pnids,
                     total=len(remaining_pnids[:limit]),
                     dynamic_ncols=True,
                     smoothing=0)
    for pnid in remaining_pnids:
        pbar.set_description(pnid)
        results.append(process_id(pnid))
        pbar.update()

    all_errors = []
    all_data = dict()
    with open("errors/MODIFIED_MODEL_WARNING.txt", "a") as model_warning_file:
        for pnid, r in results:
            if type(r) == int:
                all_errors.append((pnid, r))
                continue
            if "msg" in r and r["msg"] == "MODIFIED_MODEL":
                model_warning_file.write(f"{pnid}\n")
                del r["msg"]
            all_data[pnid] = r
    return all_data, all_errors


def process_id(pnid):
    """Create dictionary of sidechain data for a single ProteinNet ID.

    For a single ProteinNet ID i.e. ('1A9U_1_A'), fetches that PDB chain
    from the PDB and extracts its angles, coordinates, and sequence. Missing
    data is padded with GLOBAL_PAD_CHARs.
    Args:
        pnid: A ProteinNet ID, i.e. '1A9U_1_A'.

    Returns:
        Dictionary of relevant data for that ID.
    """
    message = None
    pnid_type = determine_pnid_type(pnid)
    chain = get_chain_from_proteinnetid(pnid, pnid_type)
    if not chain:
        return pnid, errors.ERRORS["NONE_STRUCTURE_ERRORS"]
    elif type(chain) == tuple and type(chain[1]) == int:
        # This indicates there was an issue parsing the chain
        return pnid, chain[1]
    elif type(chain) == tuple and chain[1] == "MODIFIED_MODEL":
        message = "MODIFIED_MODEL"
        chain = chain[0]
    try:
        dihedrals_coords_sequence = get_seq_coords_and_angles(chain)
    except errors.NonStandardAminoAcidError:
        return pnid, errors.ERRORS["NSAA_ERRORS"]
    except errors.NoneStructureError:
        return pnid, errors.ERRORS["NONE_STRUCTURE_ERRORS"]
    except errors.ContigMultipleMatchingError:
        return pnid, errors.ERRORS["MULTIPLE_CONTIG_ERRORS"]
    except errors.ShortStructureError:
        return pnid, errors.ERRORS["SHORT_ERRORS"]
    except errors.MissingAtomsError:
        return pnid, errors.ERRORS["MISSING_ATOMS_ERROR"]
    except errors.SequenceError:
        print("Not fixed.", pnid)
        return pnid, errors.ERRORS["SEQUENCE_ERRORS"]
    except ArithmeticError:
        return pnid, errors.ERRORS["NONE_STRUCTURE_ERRORS"]

    # If we've made it this far, we can unpack the data and return it
    dihedrals, coords, sequence, unmodified_seq, is_nonstd = dihedrals_coords_sequence

    if "#" not in pnid:
        try:
            dssp = PROTEIN_DSSP_DATA[pnid]
        except KeyError:
            dssp = " " * len(sequence)
    else:
        dssp = " " * len(sequence)
    resolution = get_resolution_from_pnid(pnid)
    data = {
        "ang": dihedrals,
        "crd": coords,
        "seq": sequence,
        "sec": dssp,
        "res": resolution,
        "ums": unmodified_seq,
        "mod": is_nonstd
    }
    if message:
        data["msg"] = message
    return pnid, data


def determine_pnid_type(pnid, label_astral=False):
    """Return the 'type' of a ProteinNet ID (i.e. train, valid, test, ASTRAL).

    Args:
        pnid: ProteinNet ID string.

    Returns:
        The 'type' of ProteinNet ID as a string.
    """
    if ("TBM#" in pnid or "FM#" in pnid or "TBM-hard" in pnid or "FM-hard" in pnid or
            "Unclassified" in pnid):
        return "test"

    if label_astral and pnid.count("_") == 1:
        is_astral = "_astral"
    else:
        is_astral = ""

    if "#" in pnid:
        return "valid" + is_astral
    else:
        return "train" + is_astral


def get_chain_from_trainid(pnid):
    """Return a ProDy chain object for a ProteinNet ID. Assumes train/valid ID.

    Args:
        pnid: ProteinNet ID

    Returns:
        ProDy chain object corresponding to ProteinNet ID.
    """
    modified_model_number = False
    # Try parsing the ID as a PDB ID. If it fails, assume it's an ASTRAL ID.
    try:
        pdbid, chnum, chid = pnid.split("_")
        chnum = int(chnum)
        # If this is a validation set pnid, separate the annotation from the ID
        if "#" in pdbid:
            pdbid = pdbid.split("#")[1]
    except ValueError:
        try:
            pdbid, astral_id = pnid.split("_")
            return get_chain_from_astral_id(astral_id.replace("-", "_"),
                                            ASTRAL_ID_MAPPING)
        except KeyError:
            return pnid, errors.ERRORS["MISSING_ASTRAL_IDS"]
        except (ValueError, Exception):
            return pnid, errors.ERRORS["FAILED_ASTRAL_IDS"]

    # Continue loading the chain, given the PDB ID
    use_pdb = True
    try:
        chain = pr.parsePDB(pdbid, chain=chid, model=chnum)
        if not chain:
            chain = pr.parseMMCIF(pdbid.lower(), chain=chid, model=chnum)
            use_pdb = False
    # If the file is too large, then we can download the CIF instead
    except OSError:
        try:
            chain = pr.parseMMCIF(pdbid.lower(), chain=chid, model=chnum)
            use_pdb = False
        except IndexError:
            try:
                chain = pr.parseMMCIF(pdbid.lower(), chain=chid, model=1)
                use_pdb = False
                modified_model_number = True
            except Exception as e:
                print(1, pnid, e)
                return pnid, errors.ERRORS["PARSING_ERROR_OSERROR"]
        except Exception as e:  # EOFERROR
            print(2, pnid, e)
            return pnid, errors.ERRORS["PARSING_ERROR_OSERROR"]
    except AttributeError:
        return pnid, errors.ERRORS["PARSING_ERROR_ATTRIBUTE"]
    except (pr.proteins.pdbfile.PDBParseError, IndexError):
        # For now, if the requested coordinate set doesn't exist, then we will
        # default to using the only (first) available coordinate set
        try:
            struct = pr.parsePDB(pdbid, chain=chid) if use_pdb else pr.parseMMCIF(
                pdbid.lower(), chain=chid)
        except EOFError as e:
            print(3, pnid, e)
            sys.stdout.flush()
            return pnid, errors.ERRORS["PARSING_ERROR"]
        if struct and chnum > 1:
            try:
                chain = pr.parsePDB(pdbid, chain=chid, model=1)
                modified_model_number = True
            except Exception:
                return pnid, errors.ERRORS["PARSING_ERROR"]
        else:
            return pnid, errors.ERRORS["PARSING_ERROR"]
    except Exception as e:
        print(4, pnid, e)
        return pnid, errors.ERRORS["UNKNOWN_EXCEPTIONS"]

    if chain is None:
        return pnid, errors.ERRORS["NONE_CHAINS"]

    if modified_model_number:
        return chain, "MODIFIED_MODEL"

    if contains_d_amino_acids(chain):
        return pnid, errors.ERRORS["D_AMINO_ACIDS"]

    return chain


def get_chain_from_testid(pnid):
    """Returns a ProDy chain object for a test pnid. Requires local file.

    Args:
        pnid: ProteinNet ID. Must refer to a test-set record.

    Returns:
        ProDy chain object.
    """

    category, caspid = pnid.split("#")
    try:
        chain = pr.parsePDB(os.path.join(PROTEINNET_IN_DIR, "targets", caspid + ".pdb"))
    except AttributeError:
        return pnid, errors.ERRORS["TEST_PARSING_ERRORS"]
    try:
        assert chain.numChains() == 1
    except Exception:
        print("Only a single chain should be parsed from the CASP target PDB.")
        return pnid, errors.ERRORS["TEST_PARSING_ERRORS"]
    return chain


def get_chain_from_proteinnetid(pnid, pnid_type):
    """Returns a ProDy chain for a given pnid.

    Args:
        pnid_type:
        pnid: ProteinNet ID

    Returns:
        ProDy chain object.
    """
    # If the ProteinNet ID is from the test set
    if pnid_type == "test":
        chain = get_chain_from_testid(pnid)
    # If the ProteinNet ID is from the train or validation set
    else:
        chain = get_chain_from_trainid(pnid)

    return chain


def unpack_processed_results(results, pnids):
    """Given an iterable of processed results containing angles, sequences, and PDB IDs,
    this function separates out the components (sequences as one-hot vectors, angle
    matrices, and PDB IDs) iff all were successfully preprocessed."""
    all_ohs = []
    all_angs = []
    all_crds = []
    all_ids = []
    c = 0
    for r, pnid in zip(results, pnids):
        if type(r) == int:
            # PDB failed to download
            errors.ERRORS.count(r, pnid)
            continue
        ang, coords, seq, i = r
        if no_nans_infs_allzeros(ang) and no_nans_infs_allzeros(coords):
            all_ohs.append(seq)
            all_angs.append(ang)
            all_crds.append(coords)
            all_ids.append(i)
            c += 1
    print(f"{(c * 100) / len(results):.1f}% of chains parsed. ({c}/{len(results)})")
    return all_ohs, all_angs, all_crds, all_ids


def add_proteinnetID_to_idx_mapping(data):
    """Given an already processes ProteinNet data dictionary, this function adds a mapping
    from ProteinNet ID to the subset and index number where that protein can be looked up
    in the current dictionary.

    Useful if you'd like to quickly extract a certain protein.
    """
    d = {}
    for subset in DATA_SPLITS:
        for idx, pnid in enumerate(data[subset]["ids"]):
            d[pnid] = {"subset": subset, "idx": idx}

    data["pnids"] = d
    return data


def contains_d_amino_acids(chain):
    """Return True iff the ProDy chain contains D amino acids.

    D amino acids should be excluded because their structure is not compatible
    with L amino acids and cannot be assumed to be the same.

    Args:
        chain (prody Chain): A ProDy chain representing one polypeptide molecule.
    """
    resnames = chain.getResnames()
    return any((d_aa in resnames for d_aa in D_AMINO_ACID_CODES))


def get_resolution_from_pdbid(pdbid):
    """Return RCSB-reported resolution for a PDB ID.

    Args:
        pdbid (string): RCSB PDB identifier.
    """
    query_string = ("https://data.rcsb.org/graphql?query={entry(entry_id:\"" + pdbid +
                    "\"){pdbx_vrpt_summary{PDB_resolution}}}")
    r = requests.get(query_string, headers={"User-Agent": "Mozilla/5.0"})
    if r.status_code != 200:
        res = None
    try:
        res = float(r.json()['data']['entry']['pdbx_vrpt_summary']['PDB_resolution'])
    except (KeyError, TypeError):
        res = None

    return res


def get_sequence_from_pdbid(pdbid, chain):
    """Use RSCB PDB's API to download the sequence for a PDB ID and chain.

    Args:
        pdbid (str): 4-letter PDB ID.
        chain (str): Chain name.

    Returns:
        str: 1-letter code primary sequence for the specified protein chain.
    """
    entity = 1
    query_string = (f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdbid}/{entity}")
    r = requests.get(query_string)
    if r.status_code != 200:
        res = None
    while True:
        query_string = (
            f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdbid}/{entity}")
        r = requests.get(query_string)
        entity_poly = r.json()['entity_poly']
        if chain.upper() not in entity_poly['pdbx_strand_id'].split(","):
            entity += 1
            continue
        else:
            break
    sequence = entity_poly['pdbx_seq_one_letter_code_can']

    return sequence


def get_sequence_from_pnid(pnid):
    """Return the ProteinNet ID's sequence as acquired from RCSB PDB."""
    check_for_presence_of_astral_sequence_file()

    pdbid, chid_or_astral_id, is_astral = get_pdbid_from_pnid(pnid,
                                                              return_chain=True,
                                                              include_is_astral=True)
    if is_astral:
        return get_sequence_from_astralid(chid_or_astral_id)
    return get_sequence_from_pdbid(pdbid, chid_or_astral_id)


def check_for_presence_of_astral_sequence_file():
    """Download the ASTRAL/SCOPe sequence database file from the web if not present."""
    from sidechainnet.utils.load import _download
    local_path = pkg_resources.resource_filename(
        "sidechainnet", "resources/astral-scopedom-seqres-gd-all-2.07-stable.fa")
    if not os.path.exists(local_path):
        print("Local ASTRAL sequence database not found. Downloading to", local_path)
        _download(
            "http://scop.berkeley.edu/downloads/scopeseq-2.07/astral-scopedom-seqres-gd-all-2.07-stable.fa",
            local_path)


def get_sequence_from_astralid(astral_id):
    """Read the protein sequence from the local ASTRAL/SCOPe database for an ASTRAL ID."""
    with open(
            pkg_resources.resource_filename(
                "sidechainnet", "resources/astral-scopedom-seqres-gd-all-2.07-stable.fa"),
            "r") as f:
        data = f.read()
    pattern = ">" + astral_id + r".+((\n\w+)+)"
    sequence = re.search(pattern, data, re.MULTILINE).group(1).replace("\n", "").upper()
    return sequence


def get_pdbid_from_pnid(pnid, return_chain=False, include_is_astral=False):
    """Return RCSB PDB ID associated with a given ProteinNet ID.

    Args:
        pnid (string): A ProteinNet entry identifier.
        return_chain (bool): If True, also return the chain specified by the pnid.

    Returns:
        str: The PDB ID (and chain, optional) specified by the provided ProteinNet ID.
    """
    chid = None
    is_astral = False
    # Try parsing the ID as a PDB ID. If it fails, assume it's an ASTRAL ID.
    try:
        pdbid, chnum, chid = pnid.split("_")
        chnum = int(chnum)
        # If this is a validation set pnid, separate the annotation from the ID
        if "#" in pdbid:
            pdbid = pdbid.split("#")[1]
    except ValueError:
        try:
            pdbid, astral_id = pnid.split("_")
            is_astral = True
            astral_id = astral_id.replace("-", "_")
            if "#" in pdbid:
                val_split, pdbid = pdbid.split("#")
        except Exception as e:
            print(e)
            print(pnid)
            exit(1)
    if not include_is_astral and return_chain:
        return pdbid, chid
    elif not include_is_astral:
        return pdbid
    elif include_is_astral and return_chain and not is_astral:
        return pdbid, chid, False
    elif include_is_astral and is_astral:
        return pdbid, astral_id, True
    else:
        return pdbid


def get_resolution_from_pnid(pnid):
    """Return RCSB-reported resolution for a given ProteinNet identifier."""
    if determine_pnid_type(pnid) == "test":
        return None
    return get_resolution_from_pdbid(get_pdbid_from_pnid(pnid))


def download_complete_proteinnet(user_dir=None):
    """Download and return path to complete ProteinNet (all CASPs).

    Args:
        user_dir (str, optional): If provided, download the ProteinNet data here.
            Otherwise, download it to sidechainnet/resources/proteinnet_parsed.

    Returns:
        dir_path (str): Path to directory where custom ProteinNet data was downloaded to.
    """
    from sidechainnet.utils.load import _download
    if user_dir is not None:
        dir_path = user_dir
        zip_file_path = os.path.join(user_dir, "proteinnet_parsed.zip")
    else:
        dir_path = pkg_resources.resource_filename("sidechainnet", "resources/")
        zip_file_path = pkg_resources.resource_filename(
            "sidechainnet", "resources/proteinnet_parsed.zip")

    if not os.path.isdir(os.path.join(dir_path, "proteinnet_parsed", "targets")):
        print("Downloading pre-parsed ProteinNet data (~3.5 GB compressed).")
        _download(
            "http://bits.csb.pitt.edu/~jok120/sidechainnet_data/resources/proteinnet_parsed.zip",
            zip_file_path)

        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(dir_path)
        os.remove(zip_file_path)

    else:
        print("Pre-parsed ProteinNet already downloaded.")

    dir_path = os.path.join(dir_path, "proteinnet_parsed")

    return dir_path
