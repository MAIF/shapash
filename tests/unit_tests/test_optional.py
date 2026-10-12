import warnings
from unittest.mock import patch

import pytest

from shapash._optional import import_optional_module


def test_import_optional_module_returns_imported_module():
    module = object()
    with patch("shapash._optional.importlib.import_module", return_value=module) as import_module:
        assert import_optional_module("example.module") is module
    import_module.assert_called_once_with("example.module")


def test_import_optional_module_raises_with_install_hint():
    with patch("shapash._optional.importlib.import_module", side_effect=ModuleNotFoundError):
        with pytest.raises(ImportError, match='Missing optional dependency "example". Install example'):
            import_optional_module("example", extra="Install example")


def test_import_optional_module_warns_when_missing():
    with patch("shapash._optional.importlib.import_module", side_effect=ModuleNotFoundError):
        with pytest.warns(UserWarning, match='Missing optional dependency "example". Install example'):
            assert import_optional_module("example", extra="Install example", errors="warn") is None


def test_import_optional_module_ignores_missing_dependency():
    with patch("shapash._optional.importlib.import_module", side_effect=ModuleNotFoundError):
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            assert import_optional_module("example", errors="ignore") is None
    assert not recorded
