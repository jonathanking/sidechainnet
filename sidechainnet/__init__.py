"""
SideChainNet
A protein structure prediction data set that includes side chain information. A direct extension of ProteinNet by
Mohammed AlQuraishi.
"""

# Handle versioneer
from ._version import get_versions
# Add imports here
from .structure.StructureBuilder import StructureBuilder

versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
