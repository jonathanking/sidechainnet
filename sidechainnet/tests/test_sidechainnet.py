"""Unit and regression test for the sidechainnet package."""

# Import package, test suite, and other packages as needed
import sidechainnet
import pytest
import sys


def test_sidechainnet_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "sidechainnet" in sys.modules
