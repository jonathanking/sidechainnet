"""A class to remove ugly logging code from the PyTorch Lightning Module."""

import torch


class LoggingHelper(object):
    """A simple wrapper class for obscuring uninteresting logging code."""

    def __init__(self, pl_module):
        """Create a LoggingHelper object using a PyTorch Lightning Module."""
        self.pl_module = pl_module

    def log(self, *args, **kwargs):
        self.pl_module.log(*args, **kwargs)

    def log_dict(self, *args, **kwargs):
        self.pl_module.log_dict(*args, **kwargs)

    def log_training_step(self, loss_dict, batch):
        """Log a single training step with the PyTorch Lightning Module's logger."""
        for key, value in loss_dict.items():
            if key == 'mse':
                self.log('train/rmse',
                         torch.sqrt(value),
                         on_step=True,
                         on_epoch=True,
                         prog_bar=True)
            elif key in ['angle_metrics']:
                continue
            else:
                self.log(f'train/{key}',
                         value,
                         on_step=True,
                         on_epoch=True,
                         prog_bar=True)

        self.log("trainer/batch_size", float(len(batch)), on_step=True, on_epoch=False)
        self._log_angle_metrics(loss_dict, 'train')

    def log_validation_step(self, loss_dict, dataloader_idx):
        """Log a single validation step with the PyTorch Lightning Module's logger."""
        for key, value in loss_dict.items():
            if key == 'mse':
                name = (
                    "valid/"
                    f"{self.pl_module.hparams.dataloader_name_mapping[dataloader_idx]}_rmse"
                )
                self.log(name,
                         torch.sqrt(loss_dict['mse']),
                         on_step=False,
                         on_epoch=True,
                         prog_bar=name == self.pl_module.hparams.opt_lr_scheduling_metric,
                         add_dataloader_idx=False)
            elif key in ['angle_metrics']:
                continue
            else:
                name = (
                    "valid/"
                    f"{self.pl_module.hparams.dataloader_name_mapping[dataloader_idx]}_{key}"
                )
                self.log(name,
                         value,
                         on_step=True,
                         on_epoch=True,
                         prog_bar=True,
                         add_dataloader_idx=False)

        self._log_angle_metrics(
            loss_dict, 'valid',
            self.pl_module.hparams.dataloader_name_mapping[dataloader_idx])

    def log_test_step(self, loss_dict):
        """Log a single test step with the PyTorch Lightning Module's logger."""
        for key, value in loss_dict.items():
            if key == 'mse':
                self.log('test/rmse',
                         torch.sqrt(value),
                         on_step=False,
                         on_epoch=True,
                         prog_bar=True)
            elif key in ['loss', 'angle_metrics']:
                continue
            else:
                self.log(f'test/{key}',
                         value,
                         on_step=False,
                         on_epoch=True,
                         prog_bar=True)

        self._log_angle_metrics(loss_dict, 'test')

    def _log_openmm_if_measured(self, loss_dict, split):
        """Check for presence of OpenMM loss values in loss_dict before logging."""
        # If OpenMM loss was measured, record it
        if "openmm" in loss_dict and loss_dict['openmm'] is not None:
            self.log(f"{split}/openmm",
                     loss_dict['openmm'].detach(),
                     on_step=split == 'train',
                     on_epoch=True,
                     prog_bar=True)
        # If mse_openmm loss was measured, record it
        if "mse_openmm" in loss_dict and loss_dict['mse_openmm'] is not None:
            self.log(f"{split}/openmm",
                     loss_dict['mse_openmm'].detach(),
                     on_step=split == 'train',
                     on_epoch=True,
                     prog_bar=True)

    def _log_angle_metrics(self, loss_dict, split, dataloader_name=""):
        """Log computer angle metrics."""
        dl_name_str = dataloader_name + "_" if dataloader_name else ""

        angle_metrics_dict = {
            f"{split}/{dl_name_str}{k}": loss_dict['angle_metrics'][k]
            for k in loss_dict['angle_metrics'].keys()
        }
        self.log_dict(angle_metrics_dict,
                      on_epoch=True,
                      on_step=split == 'train',
                      add_dataloader_idx=False)
