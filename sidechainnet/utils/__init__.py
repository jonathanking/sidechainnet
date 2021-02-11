"""Utility functions for SidechainNet."""
from .download import VALID_SPLITS

VALID_SPLITS_STRS = [f'valid-{s}' for s in VALID_SPLITS]
DATA_SPLITS = ['train', 'test'] + VALID_SPLITS_STRS
