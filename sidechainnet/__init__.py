"""SidechainNet.

A protein structure prediction data set that includes sidechain information. Directly
extends ProteinNet by Mohammed AlQuraishi.
"""

import os

# Handle versioneer
from ._version import get_versions

# Add imports here
from .structure.StructureBuilder import StructureBuilder
from .structure.BatchedStructureBuilder import BatchedStructureBuilder
from .utils.load import load
from .utils.download import VALID_SPLITS, DATA_SPLITS
from .utils.measure import GLOBAL_PAD_CHAR
from .create import create, create_custom, get_proteinnet_ids, generate_all
from . import utils


versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions

__author__ = "Jonathan King"
__credits__ = ("Carnegie Mellon University–"
               "University of Pittsburgh Joint PhD Program in Computational Biology\n"
               "David Koes, PhD, Advisor.")
