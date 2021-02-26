"""Utilities for minimizing SidechainNet data."""

import multiprocessing
from sidechainnet.utils.download import determine_pnid_type, get_chain_from_proteinnetid
from sidechainnet.utils.align import expand_data_with_mask

import sidechainnet as scn
from sidechainnet.openmm import openmmpdb
from tqdm import tqdm


def minimize_dict(d, parallel=True):
    """Return an OpenMM-minimized version of a SidechainNet dictionary."""
    for split in scn.DATA_SPLITS[::-1]:
        print(f"Minimizing {split}...")
        if parallel:
            d[split] = minimize_datasplit_parallel(d[split])
        else:
            d[split] = minimize_datasplit(d[split])
    return d


def minimize_datasplit(split):
    """Return an OpenMM-minimized version of a SidechainNet dictionary datasplit."""
    minimized_split = {k: [] for k in split.keys()}

    for i in tqdm(range(len(split['seq']))):
        minimized_entry = minimize_entry(split, i)
        for k in minimized_split.keys():
            minimized_split[k].append(minimized_entry[k])

    return minimized_split


def minimize_datasplit_parallel(split):
    """Return an OpenMM-minimized version of a SidechainNet dictionary datasplit."""
    minimized_split = {k: [] for k in split.keys()}

    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        results = pool.map(
            _work,
            zip(split["seq"], split["ang"], split["crd"], split["ids"], split["msk"],
                split["sec"], split["evo"], split["res"]))

    for r in results:
        for k in split.keys():
            minimized_split[k].append(r[k])

    return minimized_split


def minimize_entry(split, i):
    """Return an OpenMM-minimized version of a SidechainNet data entry."""
    new_entry = {k: None for k in split.keys()}
    sb = scn.StructureBuilder(split['seq'][i], split['crd'][i])
    sb._initialize_coordinates_and_PdbCreator()
    try:
        pdb_obj = openmmpdb.OpenMMPDB(sb.pdb_creator.get_pdb_string(), split['ids'][i])
    except ValueError as e:
        if "No template found" in str(e):
            chain = get_chain_from_proteinnetid(split['ids'][i],
                                                determine_pnid_type(split['ids'][i]))
            if isinstance(chain, tuple):
                chain = chain[0]
            raw_angs, raw_crds, seq = scn.utils.measure.get_seq_coords_and_angles(
                chain, replace_nonstd=False)
            sb = scn.StructureBuilder(seq, raw_crds)
            sb._initialize_coordinates_and_PdbCreator()
            pdb_obj = openmmpdb.OpenMMPDB(sb.pdb_creator.get_pdb_string(),
                                          split['ids'][i])

    pdb_obj.minimize_energy()

    atomgroup = pdb_obj.make_prody_atomgroup()
    angs, crds, seq = scn.utils.measure.get_seq_coords_and_angles(atomgroup)
    angs = expand_data_with_mask(angs, split['msk'][i])
    crds = expand_data_with_mask(crds, split['msk'][i])

    for k in new_entry.keys():
        if k == "ang":
            new_entry[k] = angs

            assert angs.shape == split[k][i].shape
        elif k == "crd":
            new_entry[k] = crds

            assert crds.shape == split[k][i].shape
        else:
            new_entry[k] = split[k][i]

    return new_entry


def minimize_entry_parallel(seq, ang, crd, _id, msk, sec, evo, res):
    """Return an OpenMM-minimized version of a SidechainNet data entry."""
    print(f"Minimizing {_id}... ")
    new_entry = {
        'seq': seq,
        'ids': _id,
        'msk': msk,
        'sec': sec,
        'evo': evo,
        'res': res,
        'ang': None,
        'crd': None
    }
    sb = scn.StructureBuilder(seq, crd)
    sb._initialize_coordinates_and_PdbCreator()
    try:
        pdb_obj = openmmpdb.OpenMMPDB(sb.pdb_creator.get_pdb_string(), _id)
    except ValueError as e:
        if "No template found" in str(e):
            # TODO reparse directly from a PDB file to observe non-std AAs previously ignored
            new_entry['ang'] = ang
            new_entry['seq'] = seq
            new_entry['res'] = -1
            return new_entry
    # pdb_obj = openmmpdb.OpenMMPDB(sb.pdb_creator.get_pdb_string(), _id)
    pdb_obj.minimize_energy()
    print("done.")

    atomgroup = pdb_obj.make_prody_atomgroup()
    angs, crds, _seq = scn.utils.measure.get_seq_coords_and_angles(atomgroup)
    angs = expand_data_with_mask(angs, msk)
    crds = expand_data_with_mask(crds, msk)

    assert angs.shape == ang.shape
    assert crds.shape == crd.shape
    new_entry['ang'] = angs
    new_entry['crd'] = crds

    return new_entry


def minimize_entry_coords_only(seq, coords):
    """Return an OpenMM-minimized version of SidechainNet structure (coordinates only)."""
    sb = scn.StructureBuilder(seq, coords)
    sb._initialize_coordinates_and_PdbCreator()
    pdb_obj = openmmpdb.OpenMMPDB(sb.pdb_creator.get_pdb_string())
    pdb_obj.minimize_energy()

    atomgroup = pdb_obj.make_prody_atomgroup()
    angs, crds, seq = scn.utils.measure.get_seq_coords_and_angles(atomgroup)

    return crds


def _work(seq_ang_crd__id_msk_sec_evo_res):
    """Wrap tupleized arguments for multiprocessing."""
    seq, ang, crd, _id, msk, sec, evo, res = seq_ang_crd__id_msk_sec_evo_res
    return minimize_entry_parallel(seq, ang, crd, _id, msk, sec, evo, res)


def main():
    import sidechainnet as scn
    from sidechainnet.utils import minimize
    d = scn.load("debug", "/home/jok120/sidechainnet/data/sidechainnet/")
    v = minimize.minimize_datasplit(d['valid-90'])


if __name__ == "__main__":
    main()
