import inspect
import sys

from .base_backend import BaseBackend
from .lime_backend import LimeBackend
from .shap_backend import ShapBackend


def get_backend_cls_from_name(name):
    """
    Scan current module to find the right backend with given name.
    """
    list_cls = [
        cls
        for _, cls in inspect.getmembers(sys.modules[__name__])
        if (
            inspect.isclass(cls)
            and issubclass(cls, BaseBackend)
            and cls.name.lower() == name.lower()
            and cls.name.lower() != "base"
        )
    ]

    if len(list_cls) == 1:
        return list_cls[0]
    else:
        raise ValueError(f"Backend class not found with name : {name}")
