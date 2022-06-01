"""Losses for training."""

import numpy as np
import prody as pr
import torch

from sidechainnet.structure.build_info import NUM_COORDS_PER_RES


def compute_batch_drmsd(true_coordinates, pred_coordinates, seq, verbose=False):
    """Compute batch-averaged DRMSD between two sets of coordinates.

    Args:
        true_coordinates (torch.Tensor): Tensor of true atomic coordinates
            (batch_size * length * NUM_COORDS_PER_RES).
        pred_coordinates (torch.Tensor): Tensor of predicted atomic coordinates
            (batch_size * length * NUM_COORDS_PER_RES).
        seq (torch.Tensor): Tensor of protein sequences represented as integers
            (batch_size * length).
        verbose (bool, optional): If True, print DRMSD/lnDRMSD values as they are
            computed. Defaults to False.

    Returns:
        torch.Tensor: Geometric mean of DRMSD values for each protein in the batch.
            Lower is better.
    """
    drmsds = torch.tensor(0.0)
    raw_drmsds = torch.tensor(0.0)
    for pc, tc, s in zip(pred_coordinates, true_coordinates, seq):
        # Remove batch_padding from true coords
        batch_padding = _tile((s != 20), 0, NUM_COORDS_PER_RES)
        tc = tc[batch_padding]
        missing_atoms = torch.isnan(tc).all(axis=-1)
        tc = tc[~missing_atoms]
        pc = pc[~missing_atoms]

        d = drmsd(tc, pc)
        drmsds += d / (len(tc) // NUM_COORDS_PER_RES)
        raw_drmsds += d

    if verbose:
        print(f"DRMSD = {raw_drmsds.mean():.2f}, lnDRMSD = {drmsds.mean():.2f}")
    return drmsds.mean()


def drmsd(a, b):
    """Return distance root-mean-squared-deviation between tensors a and b.

    Given 2 coordinate tensors, returns the dRMSD between them. Both
    tensors must be the exact same shape. It works by creating a mask of the
    upper-triangular indices of the pairwise distance matrix (excluding the
    diagonal). Then, the resulting values are compared with PyTorch's MSE loss.

    Args:
        a, b (torch.Tensor): coordinate tensor with shape (L x 3).

    Returns:
        res (torch.Tensor): DRMSD between a and b.
    """
    a_ = pairwise_internal_dist(a)
    b_ = pairwise_internal_dist(b)

    i = torch.triu_indices(a_.shape[0], a_.shape[1], offset=1)
    mse = torch.nn.functional.mse_loss(a_[i[0], i[1]].float(), b_[i[0], i[1]].float())
    res = torch.sqrt(mse)

    return res


def rmsd(a, b):
    """Return the RMSD between two sets of coordinates."""
    t = pr.calcTransformation(a, b)
    return pr.calcRMSD(t.apply(a), b)


def pairwise_internal_dist(x):
    """Return all pairwise distances between points in a coordinate tensor.

    An implementation of cdist (pairwise distances between sets of vectors)
    from user jacobrgardner on github. Not implemented for batches.
    https://github.com/pytorch/pytorch/issues/15253

    Args:
        x (torch.Tensor): coordinate tensor with shape (L x 3)

    Returns:
        res (torch.Tensor): a distance tensor comparing all (L x L) pairs of
                            points
    """
    x1, x2 = x, x
    assert len(x1.shape) == 2, "Pairwise internal distance method is not " \
                               "implemented for batches."
    x1_norm = x1.pow(2).sum(
        dim=-1,
        keepdim=True)  # TODO: experiment with alternative to pow, remove duplicated norm
    res = torch.addmm(x1_norm.transpose(-2, -1), x1, x2.transpose(-2, -1),
                      alpha=-2).add_(x1_norm)
    res = res.clamp_min_(1e-30).sqrt_()
    return res


def angle_mse(true, pred):
    """Return the mean squared error between two tensor batches.

    Given a predicted angle tensor and a true angle tensor (batch-padded with
    nans, and missing-item-padded with nans), this function removes all nans before
    using torch's built-in MSE loss function.

    Args:
        pred, true (np.ndarray, torch.tensor): 3 or more dimensional tensors
    Returns:
        MSE loss between true and pred.
    """
    # # Remove batch padding
    # ang_non_zero = true.ne(0).any(dim=2)
    # tgt_ang_non_zero = true[ang_non_zero]

    # Remove missing angles
    ang_non_nans = ~true.isnan()
    return torch.nn.functional.mse_loss(pred[ang_non_nans], true[ang_non_nans])


def angle_mae(true, pred):
    """Compute flattened MeanAbsoluteError between 2 angle(Rad) tensors with nan pads."""
    absolute_error = torch.abs(angle_diff(true, pred))
    return torch.nanmean(absolute_error)


def angle_diff(true, pred):
    """Compute signed distance between two angle tensors (does not change shape)."""
    error = true - pred
    # Correct for out of bounds (wrap around)
    error[error > torch.pi] -= 2 * torch.pi
    error[error < -torch.pi] += 2 * torch.pi
    return error


def _tile(a, dim, n_tile):
    # https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/4
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(
        np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)


# Structure Based Losses


def lddt_all(true, pred):
    """Compute lDDT_all between true and predicted coordinate arrays.

    See original paper for formulation: https://doi.org/10.1002/prot.23177
    Essentially the same as GDT_HA, but performed on DRMSD-like local interaction metrics.

    1. Compute all pairwise interactions.
    2. Note those that are less than 5 A away from each atom in the true structure but NOT
        from the same residue.
    3. Calculate the average fraction of interactions less than .5, 1, 2, 4 A away from
        their correct value.

    Note from the paper: "It must be noted that residues with ambiguous nomenclature for
    chemically equivalent atoms, for which the choice of atom nomenclature could influence
    the final score, were dealt with by computing a score using all possible
    nomenclatures, and by choosing the one giving the highest value. Technically, lDDT_all
    score should be computed for all."

    Args:
        true (array): True atomic coordinates. MUST contain padding/be the shape of
            (NUM_COORDS_PER_RES x L) x 3.
        pred (array): Predicted atomic coordinates. Must be the same shape as true.

    Returns:
        lddt_all_score: Value of lDDT_all.
    """
    # Compute all pairwise interactions for the true and pred structures
    true_pairwise = pairwise_internal_dist(true)  # (N x N matrix, N = number of atoms)
    pred_pairwise = pairwise_internal_dist(pred)

    def atomic_distances_of_the_same_residue_mask():
        n = len(true)  # number of atoms, must be a multiple of 14 aka heavy atom rep
        assert n // NUM_COORDS_PER_RES, "The coordinates for lDDT must be padded."
        # Make an empty mask matrix for the distance matrix
        nc = NUM_COORDS_PER_RES
        btwn_same_res_mask = torch.zeros((n, n), dtype=bool)
        # Distances between atoms in the same residue can be found in blocks along the
        # diagonal (every NUM_COORDS_PER_RES positions).
        for i in range(len(true) // NUM_COORDS_PER_RES):
            btwn_same_res_mask[i * nc:(i + 1) * nc, i * nc:(i + 1) * nc] = 1
        return btwn_same_res_mask

    # In the true structure, make a note of all interactions less than 5A away
    true_pairwise_lt5_interation_mask = true_pairwise <= 5
    # Make a mask of all the atoms that are NOT part of the same residue
    not_the_same_residue_mask = ~atomic_distances_of_the_same_residue_mask()
    # Create a mask that obscures the duplicated distances in an NxN distance matrix
    triu_mask = torch.ones(*true_pairwise.shape, dtype=torch.bool).triu(diagonal=1)

    # Select out the relevant interaction distances
    true_interactions = true_pairwise[triu_mask & true_pairwise_lt5_interation_mask &
                                      not_the_same_residue_mask]
    pred_interactions = pred_pairwise[triu_mask & true_pairwise_lt5_interation_mask &
                                      not_the_same_residue_mask]

    # Compare the interactions
    interaction_diff = torch.abs(true_interactions - pred_interactions)

    # Check if each interaction was within each threshold (4 x len(interactions))
    thresholds = torch.tensor([.5, 1, 2, 4])
    passed_check = interaction_diff <= thresholds[:, None]

    # Compute the fraction passing at each threshold
    fractions = passed_check.sum(axis=1) / passed_check.shape[1]

    # Compute lDDT_all according to the paper definition
    lddt_all_score = torch.mean(fractions)

    return lddt_all_score


def quasi_lddt_all(true, pred):
    """Compute quasi-lDDT_all between true and predicted coordinate arrays.

    Includes atomic distances between atoms of the same residue. See lddt_all for the
    more complete implementation and documentation. lddt_all excludes distances between
    atoms in the same residue. This function does not require padding.

    Args:
        true (array): True atomic coordinates. Must NOT contain padding.
        pred (array): Predicted atomic coordinates. Must be the same shape as true.

    Returns:
        lddt_all_score: Value of lDDT_all.
    """
    # Compute all pairwise interactions for the true and pred structures
    true_pairwise = pairwise_internal_dist(true)  # (N x N matrix, N = number of atoms)
    pred_pairwise = pairwise_internal_dist(pred)

    # In the true structure, make a note of all interactions less than 5A away
    true_pairwise_lt5_interation_mask = true_pairwise <= 5
    # Create a mask that obscures the duplicated distances in an NxN distance matrix
    triu_mask = torch.ones(*true_pairwise.shape, dtype=torch.bool).triu(diagonal=1)

    # Select out the relevant interaction distances
    true_interactions = true_pairwise[triu_mask & true_pairwise_lt5_interation_mask]
    pred_interactions = pred_pairwise[triu_mask & true_pairwise_lt5_interation_mask]

    # Compare the interactions
    interaction_diff = torch.abs(true_interactions - pred_interactions)

    # Check if each interaction was within each threshold (4 x len(interactions))
    thresholds = torch.tensor([.5, 1, 2, 4])
    passed_check = interaction_diff <= thresholds[:, None]

    # Compute the fraction passing at each threshold
    fractions = passed_check.sum(axis=1) / passed_check.shape[1]

    # Compute lDDT_all according to the paper definition
    lddt_all_score = torch.mean(fractions)

    return lddt_all_score


def gdt_ts(true, pred):
    """Compute GDT_TS between true (nan-padded) and predicted coordinate tensors.

    Specifically, GDT_TS should be computed over the alpha carbons only.

    Args:
        true (tensor): True atomic coordinates, must be padded with nans.
        pred (tensor): Predicted atomic coordinates.

    Returns:
        gdt_ts: Value of GDT_TS.
    """
    pass


def gdc_all(true, pred, k=10):
    """Compute GDC_ALL between true and predicted coordinate arrays.

    According to the CASP definition:
        GDC_ALL = 2*(k*GDC_P1 + (k-1)*GDC_P2 ... + 1*GDC_Pk)/(k+1)*k, k=10
        where GDC_Pk denotes percent of atoms under distance cutoff <= 0.5kÅ

        Source: https://predictioncenter.org/casp14/doc/help.html

    In my interpretation, I have modified it slightly:
        GDC_ALL = 2 * 100 *(k*GDC_P1 + (k-1)*GDC_P2 ... + 1*GDC_Pk)/((k+1)*k), k=10
        because of the desire to have it in (0, 100] and for the denominator to match
        https://doi.org/10.1016/j.heliyon.2017.e00235. The two are equivalent when k=10.

    For comparison, GDC_SC (sidechain) exists, but uses a characteristic atom for each
    sidechain (V.CG1,L.CD1,I.CD1,P.CG,M.CE,F.CZ,W.CH2,S.OG,T.OG1,C.SG,Y.OH,N.OD1,Q.OE1,
    D.OD2,E.OE2,K.NZ,R.NH2,H.NE2) instead of all atoms. See doi: 10.1002/prot.22551 for
    more discussion on GDC_SC.

    Args:
        true (array): True atomic coordinates. Must not contain padding.
        pred (array): Predicted atomic coordinates. Must be the same shape as true.

    Returns:
        gdc_all: Value of GDC_ALL.
    """
    t = pr.calcTransformation(pred, true)
    pred = t.apply(pred)

    thresholds = np.arange(1, k + 1) * 0.5
    distances = np.linalg.norm(true - pred, axis=1)

    # Check if each atom was within each threshold (len(threshold) x len(distances))
    passed_check = distances <= thresholds[:, None]

    # Compute the fraction passing at each threshold
    gdc_p = passed_check.sum(axis=1) / passed_check.shape[1]

    # Compute GDC_ALL according to the CASP definition
    gdc_all = (np.arange(1, k + 1)[::-1] * gdc_p).sum()
    gdc_all = 2 * 100 * gdc_all / ((k + 1) * k)

    return gdc_all


def tm_score(true, pred):
    """Compute TM Score between true and predicted coordinate arrays.

    See original paper for formulation: https://zhanggroup.org/TM-score/TM-score.pdf,
    DOI: 10.1002/prot.20264. Similar to GDT_TS, but is independent of protein length.

    Args:
        true (array): True atomic coordinates. Must not contain padding.
        pred (array): Predicted atomic coordinates. Must be the same shape as true.

    Returns:
        tm_score: Value of TM Score.
    """
    t = pr.calcTransformation(pred, true)
    pred = t.apply(pred)
    distances = np.linalg.norm(true - pred, axis=1)

    L = len(true)
    d0 = 1.24 * numpy_safe_cbrt(L - 15) - 1.8

    def frac(di):
        return 1 / (1 + (di / d0)**2)

    fractions = frac(distances)

    tm_score = (1 / L) * fractions.sum()

    return tm_score


def get_gdcall_tmscore(true, pred, k=10):
    """Compute GDC_ALL and TM Score for true and predicted coordinate matrices.

    Args:
        true (array): True atomic coordinates. Must not contain padding.
        pred (array): Predicted atomic coordinates. Must be the same shape as true.
        k (int, optional): Used to define bins for GDC_ALL. Defaults to 10.

    Returns:
        gdcall_tmscore (tuple): Tuple of (GDC_ALL, TM_Score).
    """
    raise NotImplementedError
    # TODO Implement a combined GDC_ALL/TM Score function to redudce duplication of effort
    # t = pr.calcTransformation(pred, true)
    # pred = t.apply(pred)

    # thresholds = np.arange(1, k + 1) * 0.5
    # distances = np.linalg.norm(true - pred, axis=1)


def numpy_safe_cbrt(a):
    """Compute the cube root of a number in a way that numpy allows.

    Numpy complains when taking the cube root of a negative number, even though it is
    defined. See https://stackoverflow.com/a/45384691/2780645.
    """
    return np.sign(a) * (np.abs(a))**(1 / 3)
