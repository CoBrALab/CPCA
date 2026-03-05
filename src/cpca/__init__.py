"""
CPCA - Complementary Principal Component Analysis

"""

__version__ = "0.1.1"

# Import main functions
from .decomposition import CPCA, cpca_modeling_
from .utils import closed_form

__all__ = [
    "CPCA",
    "cpca_modeling_",
    "closed_form",
]
