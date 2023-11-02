from sidechainnet.structure.build_info import SC_HBUILD_INFO


def test_same_number_of_bonds_angles_dihedrals():
    """
    This test makes sure that the number of angles/bonds/torsions for each
    residue is equal. In this way, a generator can be made to yield this
    information for structure generation.
    """
    for AA, AA_dict in SC_HBUILD_INFO.items():
        l = len(AA_dict["torsion-names"])
        for k in [
                "angle-vals", "bond-vals",
                "torsion-vals",  "torsion-names"
        ]:
            assert len(AA_dict[k]) == l
