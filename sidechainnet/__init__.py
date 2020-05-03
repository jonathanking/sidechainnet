"""
SideChainNet
A protein structure prediction data set that includes side chain information. A direct extension of ProteinNet by Mohammed AlQuraishi.
"""

# Add imports here
from .create import *

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
