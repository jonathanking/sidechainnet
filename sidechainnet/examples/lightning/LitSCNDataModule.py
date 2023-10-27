"""A Pytorch Lightning DataModule for SidechainNet data, downstream of scn.load()."""

import torch
import sidechainnet as scn
import pytorch_lightning as pl


class LitSCNDataModule(pl.LightningDataModule):
    """A Pytorch Lightning DataModule for SidechainNet data, downstream of scn.load()."""

    def __init__(self, scn_data_dict, batch_size, val_dataloader_target='V50'):
        """Create LitSCNDataModule from preloaded data.

        Args:
            scn_data_dict (dict): Dictionary mapping train/valid/test splits to their
                respective DataLoaders. In practice, this input is a result of calling
                scn.load(with_pytorch='dataloaders').
            val_dataloader_target (str, optional): The name of the valid dataloader to use
                as a target metric during training. Defaults to 'V50'.
        """
        super().__init__()
        self.scn_data_dict = scn_data_dict
        self.val_dataloader_idx_to_name = {}
        self.val_dataloader_target = val_dataloader_target
        self.batch_size = batch_size

    def get_train_angle_means(self, start_angle=0, end_angle=-1):
        """Return a torch tensor describing the average angle vector of the training set.

        Args:
            start_angle (int, optional): Start position in the per-residue angle array.
                Defaults to 0.
            end_angle (int, optional): End position in the per-residue angle array.
                Defaults to -1.
        """
        means = self.scn_data_dict['train'].dataset.angle_means[start_angle:end_angle]
        n_angles = means.shape[-1]
        means = torch.tensor(means).view(1, 1, len(means))
        means = scn.structure.trig_transform(means).view(1, 1, n_angles * 2).squeeze()
        return means

    def train_dataloader(self):
        """Return the SCN train set dataloader & reset descending order & batch_size."""
        self.scn_data_dict['train'].batch_sampler.turn_off_descending()
        self.scn_data_dict['train'].batch_sampler.batch_size = self.batch_size
        return self.scn_data_dict['train']

    def val_dataloader(self):
        """Return list of SCN validation set dataloaders."""
        vkeys = []
        i = 0
        for key in self.scn_data_dict.keys():
            if "valid" in key:
                renamed_key = key.replace('valid-', 'V')
                vkeys.append(key)
                self.val_dataloader_idx_to_name[i] = renamed_key
                i += 1

        # Set target validation set
        if self.val_dataloader_target not in vkeys:
            self.val_dataloader_target = vkeys[0]

        return [self.scn_data_dict[split_name] for split_name in vkeys]

    def test_dataloader(self):
        """Return SCN test set dataloader."""
        return self.scn_data_dict['test']

    def get_example_input_array(self):
        """Return a sample training batch."""
        batch = next(iter(self.train_dataloader()))
        return batch

    def set_train_dataloader_descending(self):
        """Turn on descending mode for train dataloader."""
        self.scn_data_dict['train'].batch_sampler.make_descending()
