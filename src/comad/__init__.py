"""
CoMaD - Complementary Matrix Decomposition

"""

__version__ = "0.1.1"

# Import main functions
from .decomposition import CoMaD, comad_modeling_
from .utils import closed_form

__all__ = [
    "CoMaD",
    "comad_modeling_",
    "closed_form",
]
