"""Learning rate scheduling for Transformers."""


class NoamOpt:
    """NoamOpt warmup scheduler from Alexander Rush's 'The Annotated Transformer'.

    A learning rate scheduler that implements the scheduling used in the
    Attenion is All You Need paper.

    Implementation from https://nlp.seas.harvard.edu/2018/04/03/attention.html#optimizer.
    State dict from https://stackoverflow.com/a/66696341/2780645.

    @inproceedings{opennmt,
        author    = {Guillaume Klein and
                    Yoon Kim and
                    Yuntian Deng and
                    Jean Senellart and
                    Alexander M. Rush},
        title     = {OpenNMT: Open-Source Toolkit for Neural Machine Translation},
        booktitle = {Proc. ACL},
        year      = {2017},
        url       = {https://doi.org/10.18653/v1/P17-4012},
        doi       = {10.18653/v1/P17-4012}
        }
    """

    def __init__(self, model_size, warmup, optimizer, factor=1):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size  # Think char-level size of transformer (i.e. 256)
        self._rate = 0

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @property
    def state(self):
        return self.state_dict()

    def step(self, closure=None):
        """Update parameters and rate."""
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step(closure)

    def rate(self, step=None):
        """Implement `lrate` above."""
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
             min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        """Zero gradients."""
        self.optimizer.zero_grad()

    def state_dict(self):
        """Return the state of the warmup scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Load the warmup scheduler's state.

        Arguments:
            state_dict (dict): warmup scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)
