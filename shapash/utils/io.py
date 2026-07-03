"""
IO module
"""

import datetime
import hashlib
import json
import os
import pickle
import sys
import warnings
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from typing import Any

from shapash.__version__ import __version__ as shapash_version

try:
    import yaml

    _is_yaml_available = True
except ImportError:
    _is_yaml_available = False

MANIFEST_SUFFIX = ".manifest.json"

_MODEL_FRAMEWORK_MAP = {
    "sklearn": "scikit-learn",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "catboost": "catboost",
}


def save_pickle(obj, path, protocol=pickle.HIGHEST_PROTOCOL):
    """
    Save any python Object in pickle file
    Parameters
    ----------
    obj : any Python Object
    path : str
        File path where the pickled object will be stored.
    protocol : int
        Int which indicates which protocol should be used by the pickler,
        default HIGHEST_PROTOCOL
    """

    if not isinstance(path, str):
        raise ValueError(
            """
            path parameter must be a string
            """
        )

    if not isinstance(protocol, int):
        raise ValueError(
            """
            protocol parameter must be an integer
            """
        )
    with open(path, "wb") as file:
        pickle.dump(obj, file, protocol=protocol)


def load_pickle(path):
    """
    load any pickle file
    Parameters
    ----------
    path : str
        File path where the pickled object is stored.
    Returns
    -------
    object that pickle file contains
    """

    if not isinstance(path, str):
        raise ValueError(
            """
            path parameter must be a string
            """
        )

    with open(path, "rb") as file:
        pklobj = pickle.load(file)  # noqa: S301 — caller is responsible for providing trusted paths

    return pklobj


def load_yml(path):
    """
    Loads a yml file

    Parameters
    ----------
    path : str
        File path where the yml file is stored.
    Returns
    -------
    d : dict
        Python dict containing the parsed yml file.
    """
    if _is_yaml_available is False:
        raise ModuleNotFoundError('Please install PyYAML using "pip install pyyaml" command.')

    if not isinstance(path, str):
        raise ValueError(
            """
            path parameter must be a string
            """
        )

    with open(path) as f:
        d = yaml.full_load(f)

    return d


def _try_package_version(package_name: str | None) -> str | None:
    """Return the installed version of ``package_name`` via importlib.metadata, or ``None``."""
    if not package_name:
        return None
    try:
        return _pkg_version(package_name)
    except PackageNotFoundError:
        return None


def _detect_model_framework(model: Any) -> dict:
    """
    Inspect ``model`` and return a ``{"name", "version"}`` dict describing its framework.

    Recognised frameworks: scikit-learn, xgboost, lightgbm, catboost. Anything else falls
    through with the top-level module name as the framework name.
    """
    module = type(model).__module__ or ""
    top = module.split(".")[0]
    name = _MODEL_FRAMEWORK_MAP.get(top, top or "unknown")
    version = _try_package_version(top) if top else None
    return {"name": name, "version": version}


def _compute_schema_fingerprint(predictor: Any) -> str:
    """
    Return a deterministic SHA-256 fingerprint of the predictor's user-facing schema
    (``features_dict``, ``features_types``, ``columns_dict``, ``label_dict``).

    Used at load time to detect schema tampering or stale manifests independently of
    the pickle contents.
    """
    columns_dict_str_keys = {str(k): v for k, v in (predictor.columns_dict or {}).items()}
    canonical = json.dumps(
        {
            "features_dict": predictor.features_dict,
            "features_types": predictor.features_types,
            "columns_dict": columns_dict_str_keys,
            "label_dict": predictor.label_dict,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _build_predictor_manifest(predictor: Any) -> dict:
    """Return the manifest dict describing the runtime state used to save ``predictor``."""
    return {
        "shapash_version": shapash_version,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "model_framework": _detect_model_framework(predictor.model),
        "shap_version": _try_package_version("shap"),
        "schema_fingerprint": _compute_schema_fingerprint(predictor),
        "saved_at": datetime.datetime.now(datetime.UTC).isoformat(),
    }


def _save_manifest(manifest: dict, predictor_path: str) -> str:
    """Write ``manifest`` as a sidecar JSON next to the pickle. Returns the manifest path."""
    manifest_path = predictor_path + MANIFEST_SUFFIX
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, sort_keys=True)
    return manifest_path


def _load_manifest(predictor_path: str) -> dict | None:
    """Read the sidecar manifest for ``predictor_path``. Returns ``None`` if absent."""
    manifest_path = predictor_path + MANIFEST_SUFFIX
    if not os.path.exists(manifest_path):
        return None
    with open(manifest_path, encoding="utf-8") as f:
        return json.load(f)


def _parse_major_minor(version_str: str) -> tuple | None:
    """Parse a dotted version string into ``(major, minor)``. Returns ``None`` if unparseable."""
    if not version_str:
        return None
    parts = version_str.split(".")
    try:
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
    except (ValueError, IndexError):
        return None
    return (major, minor)


def _check_predictor_manifest(manifest: dict, predictor: Any) -> None:
    """
    Validate ``manifest`` against the loaded ``predictor``. Raises ``ValueError`` on
    critical mismatches (schema fingerprint, major shapash version) and emits
    ``UserWarning`` for minor skews (minor shapash version, model framework version).
    """
    expected_fp = manifest.get("schema_fingerprint")
    actual_fp = _compute_schema_fingerprint(predictor)
    if expected_fp and expected_fp != actual_fp:
        raise ValueError(
            f"SmartPredictor schema fingerprint mismatch: manifest has {expected_fp}, "
            f"loaded predictor recomputes {actual_fp}. The pickle and manifest are out of sync "
            "(possible tampering or stale manifest)."
        )

    saved_shapash = manifest.get("shapash_version", "") or ""
    saved_v = _parse_major_minor(saved_shapash)
    current_v = _parse_major_minor(shapash_version)
    if saved_v is not None and current_v is not None:
        if saved_v[0] != current_v[0]:
            raise ValueError(
                f"SmartPredictor saved with shapash {saved_shapash}, current is {shapash_version}. "
                "Major version mismatch — re-fit and re-save the predictor."
            )
        if saved_v[1] != current_v[1]:
            warnings.warn(
                f"SmartPredictor saved with shapash {saved_shapash}, current is {shapash_version}. "
                "Minor version skew; behaviour should still be compatible.",
                UserWarning,
                stacklevel=3,
            )

    saved_fw = manifest.get("model_framework") or {}
    current_fw = _detect_model_framework(predictor.model)
    if (
        saved_fw.get("name")
        and saved_fw.get("name") == current_fw.get("name")
        and saved_fw.get("version")
        and current_fw.get("version")
        and saved_fw["version"] != current_fw["version"]
    ):
        warnings.warn(
            f"Model framework {saved_fw['name']} version differs: saved with {saved_fw['version']}, "
            f"current is {current_fw['version']}. Predictions may differ subtly.",
            UserWarning,
            stacklevel=3,
        )
