import torch
from sidechainnet.examples.losses import drmsd, rmsd


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
        self.angs_true = sc_angs_true
        self.angs_pred = sc_angs_pred

        # We know that the true angle matrix has been padded with nans.
        # Let's copy over that information into the predicted matrices.
        self.angs_pred[torch.isnan(self.angs_true)] = torch.nan

        # Init the predicted batch to be equal to the true batch but with pred angles
        self.batch_pred = self.batch_true.copy()
        for idx in range(len(batch)):
            self.batch_pred[idx].angles = self.angs_pred[idx]

        self.structures_built = False

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
    def _remove_padding(true_arr, target_arr, pad_char=0):
        """Remove padded rows (dim=-1) from a target array using a true array.

        Args:
            true_arr (tensor): True values, correctly padded with pad_char.
            target_arr (tensor): An array with the same shape of true_arr that will have
                rows removed according to the padding in true_arr.
            pad_char (int, optional): Padding character used in true_array. Defaults to 0.

        Returns:
            tensor: A version of target_arr where any padded rows have been removed. May
                not match the same size.
        """
        real_valued_rows = (true_arr != pad_char).any(dim=-1)
        return target_arr[real_valued_rows]

    def map_over_coords(self, fn, remove_padding=True, len_normalize=False):
        """Return a list of values after mapping a function over the (true, pred) coords.

        Args:
            fn (function): A function that takes two arguments (protein coordinates) and
                returns a value.
            remove_padding (bool, optional): If True, remove padding from coordinate
                vectors. Defaults to True.
            len_normalize (bool, optional): If True, scale the result of the function by
                the length of each protein. Defaults to False.

        Returns:
            list: A list of computed values.
        """
        if not self.structures_built:
            self.build_structures_from_angles()

        def compute_values():
            for t, p in zip(self.batch_true, self.batch_pred):
                if remove_padding:
                    a = self._remove_padding(t.coords, t.coords)
                    b = self._remove_padding(t.coords, p.coords)
                else:
                    a = t.coords
                    b = p.coords
                value = fn(a, b) if not len_normalize else fn(a, b) / len(t)
                yield value

        values = compute_values()

        return list(values)

    def rmsd(self):
        return torch.mean(self.map_over_coords(rmsd))

    def drmsd(self):
        return torch.mean(self.map_over_coords(drmsd))

    def lndrmsd(self):
        return torch.mean(self.map_over_coords(drmsd, len_normalize=True))
