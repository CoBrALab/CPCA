"""
CPCA - Complementary Principal Component Analysis

"""

__version__ = "0.1.0"

# Import main functions from modeling
from .modeling import cpca, cpca_auto, cpca_quick, spatial_cpca
# Import report functions if needed
from .report import (cosine_similarity, evaluate_fit, gen_report, optim_n,
                     plot_report)

__all__ = [
    "spatial_cpca",
    "cpca",
    "cpca_quick",
    "cpca_auto",
    cosine_similarity,
    gen_report,
    evaluate_fit,
    optim_n,
    plot_report,
]
