import numpy as np


class ScheduledOptim():
    """
    A learning rate scheduler that implements the scheduling used in the
    Attenion is All You Need paper. Work extends on original implemention by
    github user jadore801120.
    https://github.com/jadore801120/attention-is-all-you-need-pytorch
    """

    def __init__(self, optimizer, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)

    def step(self):
        """
        Update the base optimizer.
        """
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        """
        Zero gradients.
        """
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        """
        Update learning rate per policy described in original paper.
        Linear ramp-up, exponential decay.
        """

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()
        self.cur_lr = lr
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

    def state_dict(self):
        """
        Returns a tuple of the relevant state information for saving this
        optimizer.
        """
        return (self._optimizer.state_dict(), self.n_warmup_steps, self.n_current_steps, self.init_lr)

    @property
    def state(self):
        return self._optimizer.state_dict()

    def load_state_dict(self, d):
        """
        Loads the state information for a saved version of this optimizer.
        """
        self._optimizer.load_state_dict(d[0])
        self.n_warmup_steps = d[1]
        self.n_current_steps = d[2]
        self.init_lr = d[3]
