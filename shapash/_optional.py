import importlib
import warnings
from typing import Literal


def import_optional_module(module_name: str, extra: str = "", errors: Literal["raise", "warn", "ignore"] = "raise"):
    """Import an optional dependency by module name.

    This helper tries to import the requested module using importlib.
    If the module is not installed, behavior depends on the ``errors`` mode:

    - ``raise``: raise a clean ``ImportError`` with an explanatory message.
    - ``warn``: emit a ``UserWarning`` and return ``None``.
    - ``ignore``: return ``None`` silently.

    Parameters
    ----------
    module_name : str
        Name of the module to import, e.g. ``"shap"`` or ``"pandas"``.
    extra : str, optional
        Additional text appended to the error or warning message.
        This can be used to include install instructions or fallback hints.
    errors : {"raise", "warn", "ignore"}, default ``"raise"``
        How to handle a missing module.

    Returns
    -------
    module | None
        The imported module if available, otherwise ``None`` when missing and
        ``errors`` is ``"warn"`` or ``"ignore"``.

    Raises
    ------
    ImportError
        If the module is missing and ``errors`` is ``"raise"``.
    """
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        msg = f'Missing optional dependency "{module_name}". {extra}'.strip()
        if errors == "raise":
            raise ImportError(msg) from None
        if errors == "warn":
            warnings.warn(msg, UserWarning, stacklevel=2)
        return None
