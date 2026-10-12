from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("shapash")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0+unknown"

VERSION = tuple(int(p) for p in __version__.split(".")[:3] if p.isdigit())
