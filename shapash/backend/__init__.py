import inspect
import sys

from .backend import Backend
from .base_backend import BaseBackend
from .lime_backend import LimeBackend
from .nlp_backend import NlpBackend, NlpContributions
from .nlp_captum_lig_backend import NlpCaptumLigBackend
from .nlp_lime_backend import NlpLimeBackend
from .nlp_shap_backend import NlpShapBackend
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
            and issubclass(cls, Backend)
            and cls.name.lower() == name.lower()
            and cls.name.lower() != "base"
        )
    ]

    if len(list_cls) == 1:
        return list_cls[0]
    else:
        raise ValueError(f"Backend class not found with name : {name}")
