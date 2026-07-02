"""Public package exports for scuva."""

from .text import wrap_join, rename, clean_title
from .graphing import set_categorical_colors, get_categorical_colormap, subplots_with_legend_axis
from .graphing.legend import make_colorbar, make_legend
from .graphing.umap import multiple_umap, umap, umap_split
from .graphing.composition import graph_counts, graph_proportions
from .graphing.qc_filtering import qc_metrics_violinplot, qc_metrics_scatterplot
from .analysis.expression_by_obs import get_expression_by_obs, check_expression
from .analysis.qc_filtering import is_outlier

__version__ = "0.1.2-a2"

__all__ = [
    "__version__", 
    "is_outlier",
    "qc_metrics_violinplot",
    "qc_metrics_scatterplot",
    "umap", 
    "multiple_umap", 
    "umap_split",
    "graph_counts",
    "graph_proportions",
    "get_expression_by_obs",
    "check_expression",
    "make_colorbar", 
    "make_legend",
    "wrap_join",
    "rename",
    "clean_title",
    "set_categorical_colors",
    "get_categorical_colormap",
    "subplots_with_legend_axis",
]
