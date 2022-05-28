"""A class that helps manipulate a set of angle structure predictions for analysis."""
import numpy as np
import torch
from sidechainnet.examples.losses import angle_diff, angle_mse, drmsd, rmsd
from sidechainnet.structure.build_info import ANGLE_IDX_TO_NAME_MAP, NUM_SC_ANGLES
from sidechainnet.structure.structure import inverse_trig_transform
from sidechainnet.utils.measure import GLOBAL_PAD_CHAR
from sidechainnet.utils.openmm_loss import OpenMMEnergyH


class AnglePredictionHelper(object):
    """Helps analyze protein structure predictions via torsional angles."""

    def __init__(self, batch, sc_angs_true, sc_angs_pred) -> None:
        """Initialize AnglePredictionHelper.

        Args:
            batch (ProteinBatch): A ProteinBatch object yielded during training.
            sc_angs_true (tensor): nan-padded tensor of true angles.
            sc_angs_pred (tensor): real-valued tensor of predicted protein angles.
        """
        self.batch_true = batch
        self.batch_pred = None
        self.sc_angs_true = sc_angs_true
        self.sc_angs_pred = sc_angs_pred.clone()  # A differentiable copy

        # We know that the true angle matrix has been padded with nans.
        # Let's copy over that information into the predicted matrices.
        assert torch.isnan(self.sc_angs_true).any(), ("The true angle tensor must be "
                                                      "padded with nans (0 found).")
        self.sc_angs_pred[torch.isnan(self.sc_angs_true)] = torch.nan

        # Convert sin/cos values into radians
        self.sc_angs_pred_rad = inverse_trig_transform(self.sc_angs_pred, n_angles=6)
        self.sc_angs_true_rad = inverse_trig_transform(self.sc_angs_true, n_angles=6)

        # Init the predicted batch to start as a copy of the true batch
        self.batch_pred = self.batch_true.copy()

        # Move coordinate/angle numpy arrays -> torch tensors
        self.batch_pred.torch()
        self.batch_true.torch()

        # However, we add the predicted data (sc-angles) to the batch explicitly
        for idx in range(len(batch)):
            # We must also slice the angle tensor to remove sequence length padding
            pred_angles_rad = self.sc_angs_pred_rad[idx, 0:len(self.batch_true[idx])]
            self.batch_pred[idx].angles[:, -NUM_SC_ANGLES:] = pred_angles_rad

        self.structures_built = False
        self._unpadded_coords = []

    def build_structures_from_angles(self):
        """Call build_coords_from_angles on each underlying SCNProtein for comparison."""
        # Build true structures
        for p in self.batch_true:
            assert p.sb is None, "Protein has already been built."
            p.build_coords_from_angles()
            # TODO Note that this builds with 0 padding, why is this default?

        # Build predicted structures
        for p in self.batch_pred:
            assert p.sb is None, "Protein has already been built."
            p.build_coords_from_angles()

        self.structures_built = True
        # TODO may be parallelized

    @staticmethod
    def _remove_padding(true_arr, target_arr, pad_char=GLOBAL_PAD_CHAR):
        """Remove padded rows (dim=-1) from a target array using a true array.

        In this context, this function is used to remove empty atom rows from coordinate
        tensors. SidechainNet uses an atomic representation with an equal number of atoms
        per residue. Empty atom rows are typically representing using tensor([nan]*3).

        Args:
            true_arr (tensor): True values, correctly padded with pad_char.
            target_arr (tensor): An array with the same shape of true_arr that will have
                rows removed according to the padding in true_arr.
            pad_char (int, optional): Padding character used in true_array. Defaults to 0.

        Returns:
            tensor: A version of target_arr where any padded rows have been removed. May
                not match the same size.
        """
        if np.isnan(pad_char):
            real_valued_rows = (~torch.isnan(true_arr)).any(dim=-1)
        else:
            real_valued_rows = (true_arr != pad_char).any(dim=-1)
        return target_arr[real_valued_rows]

    def map_over_coords(self,
                        fn,
                        remove_padding=True,
                        len_normalize=False,
                        make_numpy=False):
        """Return a list of values after mapping a function over the (true, pred) coords.

        Args:
            fn (function): A function that takes two arguments (protein coordinates) and
                returns a value.
            remove_padding (bool, optional): If True, remove padding from coordinate
                vectors. Defaults to True.
            len_normalize (bool, optional): If True, scale the result of the function by
                the length of each protein. Defaults to False.
            make_numpy (bool, optional): If True, convert the coordinate tensors to numpy
                matrices before executing the function.

        Returns:
            list: A list of computed values.
        """
        if not self.structures_built:
            self.build_structures_from_angles()

        def compute_values():
            for i, (t, p) in enumerate(zip(self.batch_true, self.batch_pred)):
                if remove_padding and len(self._unpadded_coords) < len(self.batch_true):
                    self._unpadded_coords = []
                    a = self._remove_padding(t.coords, t.coords)
                    b = self._remove_padding(t.coords, p.coords)
                    self._unpadded_coords.append((a, b))
                elif remove_padding:
                    a, b = self._unpadded_coords[i]
                else:
                    a = t.coords
                    b = p.coords

                if make_numpy:
                    a = a.detach().numpy()
                    b = b.detach().numpy()

                value = fn(a, b) if not len_normalize else fn(a, b) / len(t)
                yield value

        values = compute_values()

        return torch.as_tensor(list(values))

    def rmsd(self):
        """Compute RMSD over all pairs of coordinates in the predicted batch."""
        # We use `make_numpy` because ProDy cannot operate on tensors
        rmsd_val = torch.mean(self.map_over_coords(rmsd, make_numpy=True))
        return rmsd_val

    def drmsd(self):
        """Compute DRMSD over all pairs of coordinates in the predicted batch."""
        return torch.mean(self.map_over_coords(drmsd))

    def lndrmsd(self):
        """Compute len-normalized DRMSD over all coord pairs in the predicted batch."""
        return torch.mean(self.map_over_coords(drmsd, len_normalize=True))

    def openmm_loss(self):
        """Compute OpenMMEnergyH loss for predicted structures."""
        # TODO make more efficient using persisent batch_pred / prebuilt structures
        return self._compute_openmm_loss(self.batch_true, self.sc_angs_pred)

    def angle_mse(self):
        """Return the mean squared error between two angle tensors padded with nans."""
        return angle_mse(self.sc_angs_true, self.sc_angs_pred)

    def angle_metrics_dict(self):
        """Return a dictionary containing a large number of relevant angle metrics."""
        empty_dict = {}
        return self._compute_angle_metrics(self.sc_angs_true, self.sc_angs_pred,
                                           empty_dict)

    def structure_metrics_dict(self):
        """Return a dictionary containing a large number of relevant structure metrics."""
        empty_dict = {}
        return self._compute_angle_metrics(self.sc_angs_true, self.sc_angs_pred,
                                           empty_dict)

    def _compute_structure_metrics(self, batch, sc_angs_true, sc_angs_pred, loss_dict):
        """Compute & return loss dict with structure-based performance metrics (ie rmsd).

        Args:
            batch (ProteinBatch): A ProteinBatch object yielded during training.
            sc_angs_true (tensor): nan-padded tensor of true angles.
            sc_angs_pred (tensor): real-valued tensor of predicted protein angles.
            loss_dict (dict): Dictionary whose keys are metrics/losses to be recorded.
        """
        loss_dict['rmsd'] = self.rmsd()
        loss_dict['drmsd'] = self.drmsd()
        loss_dict['lndrmsd'] = self.lndrmsd()
        return loss_dict

    def _compute_angle_metrics(self,
                               sc_angs_true,
                               sc_angs_pred,
                               loss_dict,
                               acc_tol=np.deg2rad(20)):
        """Compute MAE(X1-5) and accuracy (correct within 20 deg).

        Args:
            sc_angs_true (torch.Tensor): True sidechain angles, padded with nans.
            sc_angs_pred (torch.Tensor): Predicted sidechain angles.
            loss_dict (dict): Dictionary for loss and metric record keeping.

        Returns:
            loss_dict (dict): Updated dictionary containing new keys MAE_X1...ACC.
        """
        # Absolue Error Only (contains nans)
        abs_error = torch.abs(
            angle_diff(self.sc_angs_true_rad.detach(), self.sc_angs_pred_rad.detach()))

        loss_dict['angle_metrics'] = {}

        # Compute MAE by angle; Collapse sequence dimension and batch dimension
        mae_by_angle = torch.nanmean(abs_error, dim=1).mean(dim=0)
        assert len(mae_by_angle) == 6, "MAE by angle should have length 6."
        for i in range(6):
            angle_name_idx = i + 6
            angle_name = ANGLE_IDX_TO_NAME_MAP[angle_name_idx]
            cur_mae = mae_by_angle[i]
            if torch.isnan(cur_mae):
                continue
            loss_dict["angle_metrics"][f"{angle_name}_mae_rad"] = cur_mae
            loss_dict["angle_metrics"][f"{angle_name}_mae_deg"] = torch.rad2deg(cur_mae)

        # Compute angles within tolerance (contains nans)
        correct_angle_preds = abs_error < acc_tol

        # An entire residue is correct if the residue has >= 1 predicted angles
        # and all of those angles are correct. When computing correctness, an angle is
        # correct if it met the tolerance or if it is nan.
        res_is_predicted = ~torch.isnan(abs_error).all(dim=-1)  # Avoids all-nan rows
        res_is_correct = torch.logical_or(correct_angle_preds,
                                          torch.isnan(abs_error)).all(dim=-1)
        res_is_predicted_and_correct = torch.logical_and(res_is_predicted, res_is_correct)
        accuracy = res_is_predicted_and_correct.sum() / res_is_predicted.sum()

        loss_dict["angle_metrics"]["acc"] = accuracy

        # Compute Accuracy per Angle (APA) which evaluates each angle indp.
        ang_is_predicted = ~torch.isnan(correct_angle_preds)
        ang_is_correct = torch.logical_or(correct_angle_preds, torch.isnan(abs_error))
        num_correct_angles = torch.logical_and(ang_is_predicted, ang_is_correct).sum()
        apa = num_correct_angles / ang_is_predicted.sum()
        loss_dict['angle_metrics']['apa'] = apa

        return loss_dict

    def _compute_openmm_loss(self, batch, sc_angs_pred):
        # TODO ignore protein entries like 1QCR_9_I that are a-carbon only
        proteins = batch
        loss_total = count = 0.
        sc_angs_pred_rad = inverse_trig_transform(sc_angs_pred, n_angles=6)
        loss_fn = OpenMMEnergyH()
        for p, sca in zip(proteins, sc_angs_pred_rad):
            p.cuda()
            # Fill in protein obj with predicted angles instead of true angles
            if "-" in p.mask:
                continue
            p.trim_to_max_seq_len()  # TODO implement update_with_pred_angs
            p.angles[:, 6:] = sca[:len(p)]  # we truncate here to remove batch pa
            p.add_hydrogens(from_angles=True)
            eloss = loss_fn.apply(p, p.hcoords)
            if ~torch.isfinite(eloss):
                continue
            loss_total += eloss
            count += 1

        if count:
            loss = loss_total / count
        else:
            loss = None
        return loss
