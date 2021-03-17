"""SidechainNet.

A protein structure prediction data set that includes sidechain information. Directly
extends ProteinNet by Mohammed AlQuraishi.
"""

import os

_ROOT = os.path.abspath(os.path.dirname(__file__))


def get_data(path):
    """Return absolute path to specified package resource.

    Args:
        path (str): Filename of resource, e.g. "astral_data.txt".

    Returns:
        str: Path to requested resource.
    """
    return os.path.join(_ROOT, 'resources', path)


# Handle versioneer
from ._version import get_versions

# Add imports here
from .structure.StructureBuilder import StructureBuilder
from .structure.BatchedStructureBuilder import BatchedStructureBuilder
from .utils.load import load

from .utils.download import VALID_SPLITS, DATA_SPLITS


versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions

__author__ = "Jonathan King"
__credits__ = (
    "Carnegie Mellon University–"
    "University of Pittsburgh Joint PhD Program in Computational Biology\n"
    "David Koes, PhD, Advisor."
)
