# dynamic versioning
from importlib.metadata import version, PackageNotFoundError
from sen2sr.utils import predict_large
from sen2sr.xai.lam import lam

try:
    __version__ = version("sen2sr")
except PackageNotFoundError:    
    __version__ = "unknown"


try:
    import torch
    import timm
except ImportError:
    raise ImportError(
        "sen2sr requires torch and timm. Please install them."        
    )

__all__ = [
    "__version__",
    "predict_large",
    "lam"
]