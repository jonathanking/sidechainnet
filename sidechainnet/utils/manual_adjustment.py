from sidechainnet.utils.align import binary_mask_to_str


def manually_correct_mask(pnid, pn_entry, mask):
    if pnid == "3TDN_1_A":
        # In this case, the default mask from ProteinNet was actually correct. The
        # protein's sequence has two equal scoring alignments, but the aligners typically
        # pick the "incorrect" one.
        mask = binary_mask_to_str(pn_entry['mask'])
    return mask


def needs_manual_adjustment(pnid):
    """Declares a list of pnids that should be handled manually due to eggregious
    differences between observed and expected seqeuences and masks. """
    if pnid in [
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
            "4BDL_1_A", "3J9M_79_AY"
    ]:
        return True
    else:
        return False