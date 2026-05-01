"""Public package exports for scuva."""

__version__ = "0.1.0rc1"

from .text import wrap_join, rename, clean_title
from .graphing import set_categorical_colors, get_categorical_colormap, subplots_with_side_axis
from .graphing.legend import make_colorbar, make_legend
from .graphing.umap import multiple_umap, umap, umap_split
from .graphing.composition import graph_counts, graph_proportions

__all__ = [
    "__version__", 
    "umap", 
    "multiple_umap", 
    "umap_split",
    "graph_counts",
    "graph_proportions",
    "make_colorbar", 
    "make_legend",
    "wrap_join",
    "rename",
    "clean_title",
    "set_categorical_colors",
    "get_categorical_colormap",
    "subplots_with_side_axis",
]
