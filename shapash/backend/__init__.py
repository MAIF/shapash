import sys
import inspect

from .base_backend import BaseBackend
from .shap_backend import ShapBackend
from .acv_backend import AcvBackend
from .lime_backend import LimeBackend


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
                and cls.name.lower() != 'base'
        )
    ]

    if len(list_cls) == 1:
        return list_cls[0]
    else:
        raise ValueError(f"Backend class not found with name : {name}")
