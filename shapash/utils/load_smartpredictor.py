"""
load_smartpredictor module
"""

import warnings

from shapash.explainer.smart_predictor import SmartPredictor
from shapash.utils.io import _check_predictor_manifest, _load_manifest, load_pickle


def load_smartpredictor(path: str) -> SmartPredictor:
    """
    load_smartpredictor allows Shapash users to load SmartPredictor Object already saved into a pickle.

    Parameters
    ----------
    path : str
        File path of the pickle file.

    Example
    --------
    >>> predictor = load_smartpredictor('path_to_pkl/predictor.pkl')

    Notes
    -----
    If a sidecar manifest ``path + ".manifest.json"`` is present (predictors saved
    with shapash >= 2.10), provenance checks are run: the schema fingerprint and the
    shapash major version must match the running environment, otherwise a
    ``ValueError`` is raised. Minor shapash and model-framework version skews emit a
    ``UserWarning``. Predictors saved without a manifest still load, with a
    ``DeprecationWarning``.
    """
    predictor = load_pickle(path)
    if not isinstance(predictor, SmartPredictor):
        raise ValueError(f"{predictor} is not an instance of type SmartPredictor")

    manifest = _load_manifest(path)
    if manifest is None:
        warnings.warn(
            f"Loading SmartPredictor from {path} without a manifest sidecar. "
            "Provenance checks (shapash version, schema fingerprint) are skipped. "
            "Re-save the predictor to generate a manifest.",
            DeprecationWarning,
            stacklevel=2,
        )
    else:
        _check_predictor_manifest(manifest, predictor)
    return predictor
