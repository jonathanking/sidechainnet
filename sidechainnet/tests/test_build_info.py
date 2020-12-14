from sidechainnet.structure.build_info import SC_BUILD_INFO


def test_same_number_of_bonds_angles_dihedrals():
    """
    This test makes sure that the number of angles/bonds/torsions for each
    residue is equal. In this way, a generator can be made to yield this
    information for structure generation.
    """
    for AA, AA_dict in SC_BUILD_INFO.items():
        l = len(AA_dict["angles-names"])
        for k in [
                "bonds-names", "torsion-names", "angles-vals", "bonds-vals",
                "torsion-vals", "bonds-types", "angles-types", "torsion-types"
        ]:
            assert len(AA_dict[k]) == l
