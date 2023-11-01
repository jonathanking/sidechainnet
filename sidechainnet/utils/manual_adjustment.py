"""A module for the handling of specific, problematic ProteinNet entries."""

from sidechainnet.structure.build_info import NUM_COORDS_PER_RES
import copy


NEEDS_MANUAL_ADJUSTMENT = [
        "4PGI_1_A", "3CMG_1_A", "4ARW_1_A", "4Z08_1_A", "2PLV_1_1", "4PG7_1_A",
        "2O24_1_A", "5I4N_1_A", "4RYK_1_A", "1CS4_3_C", "3SRY_1_A", "2AV4_1_A",
        "3GW7_1_A", "1TQ5_1_A", "5DND_1_A", "4YCU_1_A", "1VRZ_1_A", "1RRX_1_A",
        "2XUV_1_A", "2CFO_1_A", "5DNC_1_A", "2WTS_1_A", "4JQI_3_L", "2H9W_1_A",
        "5DNE_1_A", "3RN8_1_A", "4RQF_2_A", "2FLQ_1_A", "3IPN_1_A", "3GP3_1_A",
        "2Q6P_1_A", "4O2D_1_A", "2XXX_1_A", "3AB4_2_B", "2PLV_1_1", "4UQQ_1_A",
        "2DTJ_1_A", "4ORN_1_A", "4PG7_1_A", "2XXR_1_A", "3IG5_1_A", "3FPH_1_A",
        "2O24_1_A", "4RQE_1_A", "2LIG_1_A", "4XMR_1_A", "1CT9_1_A", "1KL1_1_A",
        "3Q1X_1_A", "1II5_1_A", "2XII_1_A", "3SRY_1_A", "4YCW_1_A", "3ZDQ_1_A",
        "1YJS_1_A", "4CVK_1_A", "2VSQ_1_A", "3P47_1_A", "4D57_1_A", "3WVN_1_A",
        "2XXU_1_A", "3VSC_1_A", "3S1T_1_A", "2AV4_1_A", "3RNN_1_A", "1WNU_1_A",
        "4BDL_1_A", "3J9M_79_AY",
    ]

# IDs that are inappropriate for training as of November 1, 2023.
NEEDS_NEW_ADJUSTMENT = ["3JAJ_49_2", "2I2J_1_A",]


def manually_correct_mask(pnid, pn_entry, mask):
    """Corrects a protein sequence mask for a given ProteinNet ID."""
    from sidechainnet.utils.align import binary_mask_to_str
    if pnid == "3TDN_1_A":
        # In this case, the default mask from ProteinNet was actually correct. The
        # protein's sequence has two equal scoring alignments, but the aligners typically
        # pick the "incorrect" one.
        mask = binary_mask_to_str(pn_entry['mask'])
    return mask


def needs_manual_adjustment(pnid):
    """Declares a list of pnids that should be handled manually due to eggregious
    differences between observed and expected seqeuences and masks."""
    return pnid in NEEDS_MANUAL_ADJUSTMENT + NEEDS_NEW_ADJUSTMENT


def remove_problematic_pnids_from_scndict(raw_data):
    """Remove problematic pnids from a scndict.

    Args:
        raw_data (dict): SidechainNet dictionary.

    Returns:
        Filtered dictionary.
    """
    for split in raw_data.keys():
        if "train" not in split or "valid" not in split or "test" not in split:
            continue
        new_split_data = {
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
        this_split = raw_data[split]
        for seq, ang, crd, msk, evo, _id, res, sec, ums, mod in zip(
                this_split['seq'], this_split['ang'], this_split['crd'],
                this_split['msk'], this_split['evo'], this_split['ids'],
                this_split['res'], this_split['sec'], this_split['ums'],
                this_split['mod']):
            if needs_manual_adjustment(_id):
                continue
            else:
                new_split_data["seq"].append(seq)
                new_split_data["ang"].append(ang)
                new_split_data["ids"].append(_id)
                new_split_data["evo"].append(evo)
                new_split_data["msk"].append(msk)
                new_split_data["crd"].append(crd)
                new_split_data["sec"].append(sec)
                new_split_data["res"].append(res)
                new_split_data["ums"].append(ums)
                new_split_data["mod"].append(mod)

        raw_data[split] = new_split_data
    return raw_data


def manually_adjust_data(pnid, sc_entry):
    """Returns a modified version of sc_entry to fix some issues manually.

    Args:
        pnid: string, ProteinNet ID
        sc_entry: dictionary containing "seq", "ang", "crd" data

    Returns:
        If sc_entry must be modified, then it is corrected and returned.
        Otherwise, it is returned without modifications.
    """

    # In the case of 5FXN, ProDy mistakenly parses two extraneous residues "VK"
    # from the file due to the file not distinguishing these resides as being
    # on a different segment. We can manually remove this data here before
    # proceeding with alignment.
    # https://github.com/prody/ProDy/issues/1045
    if "5FXN" in pnid and len(sc_entry["seq"]) == 316 and sc_entry["seq"][-3:] == "VVK":
        sc_entry["seq"] = sc_entry["seq"][:-2]
        sc_entry["ang"] = sc_entry["ang"][:-2]
        sc_entry["crd"] = sc_entry["crd"][:-NUM_COORDS_PER_RES * 2]

    return sc_entry


def correct_1GJJ_A_1(data):
    """Correct the sidechain data for 1GJJ_A_1, which has two separate chains.

    The file uploaded to RCSB PDB contains two overlapping domains so a manual adjustment
    is required. See https://github.com/jonathanking/sidechainnet/issues/38 for more
    details.
    """
    partA = copy.deepcopy(data)
    partB = copy.deepcopy(data)
    for key in data.keys():
        if key == 'ums':
            res_list = data[key]
            partA[key] = res_list[0:50]
            partB[key] = res_list[50:]
        elif key == 'crd':
            partA[key] = partA[key][0:50 * NUM_COORDS_PER_RES]
            partB[key] = partB[key][50 * NUM_COORDS_PER_RES:]
        elif key == 'res':
            continue
        else:
            partA[key] = partA[key][0:50]
            partB[key] = partB[key][50:]

    return partA, partB
