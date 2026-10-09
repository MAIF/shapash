import importlib
import sys
from importlib.metadata import PackageNotFoundError
from unittest.mock import patch

import pytest

import shapash  # noqa: F401

# ``shapash.__version__`` resolves to the version string, not the submodule.
version_module = sys.modules["shapash.__version__"]


@pytest.fixture
def reload_version_module():
    """Re-execute ``shapash.__version__`` under a test patch, then restore the real values."""
    yield lambda: importlib.reload(version_module)
    importlib.reload(version_module)


def test_version_reads_package_metadata(reload_version_module):
    with patch("importlib.metadata.version", return_value="2.10.1"):
        reload_version_module()
    assert version_module.__version__ == "2.10.1"
    assert version_module.VERSION == (2, 10, 1)


def test_version_falls_back_when_not_installed(reload_version_module):
    with patch("importlib.metadata.version", side_effect=PackageNotFoundError("shapash")):
        reload_version_module()
    assert version_module.__version__ == "0+unknown"
    assert version_module.VERSION == ()
