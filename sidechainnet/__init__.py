"""
SideChainNet
A protein structure prediction data set that includes sidechain information. Directly 
extends ProteinNet by Mohammed AlQuraishi.
"""

# Handle versioneer
from ._version import get_versions

# Add imports here
from .structure.StructureBuilder import StructureBuilder
from .structure.BatchedStructureBuilder import BatchedStructureBuilder
from .utils.load import load

versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
