"""Shared plotting utilities for categorical colors and figure layout."""

from typing import Literal

from matplotlib import colors as mplc
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from anndata import AnnData
from pandas import CategoricalDtype
from scanpy.plotting.palettes import default_20, default_28, default_102

DEFAULT_CMAP = mplc.LinearSegmentedColormap.from_list("gray_blue_yellow", colors=["lightgray", "royalblue", "gold"])
DPI = 300

def set_categorical_colors(
    adata: AnnData,
    feature: str,
    color_mapping: dict[str, str]
):
    """Set or override colors for a categorical observation column.

    Parameters
    ----------
    adata
        AnnData object containing the categorical column.
    feature
        Name of the categorical column in ``adata.obs``.
    color_mapping
        Mapping from category label to a Matplotlib-compatible color string.

    Notes
    -----
    Colors are stored in ``adata.uns[f"{feature}_colors"]`` using the standard
    Scanpy convention. Any categories not included in ``color_mapping`` keep their
    existing colors or receive default Scanpy palette colors.
    """
    if feature not in adata.obs.columns:
        raise ValueError(f"'{feature}' not found in adata.obs")
    if not isinstance(adata.obs[feature].dtype, CategoricalDtype):
        raise ValueError(
            f"'{feature}' is not categorical data. Check that the column values "
            + "are correct before casting to categorical")
    if len(color_mapping) == 0:
        print("No categories provided, no colors were changed.")
        return
    non_str_keys = [k for k in color_mapping.keys() if not isinstance(k, str)]
    if len(non_str_keys) > 0:
        raise TypeError(
            "All keys in the colors dict must be strings. "
            + "Categorical data that appears numeric, such as clusters, "
            + "should be represented as a string: '0', '13', etc. Non-string keys found: "
            + ", ".join(str(k) for k in non_str_keys)
        )
        
    cats = adata.obs[feature].cat.categories
    nonexistant_keys = [k for k in color_mapping.keys() if k not in cats]
    if len(nonexistant_keys) > 0:
        raise ValueError(
            "Some category keys were provided that aren't present in the categorical column: "
            + ", ".join(k for k in nonexistant_keys)
        )
    
    missing_cats = [c for c in cats if c not in color_mapping]
    if len(missing_cats) > 0:
        print("Some colors were not provided, using previously set or default colors: " + ", ".join(missing_cats))
    
    ckey = feature + "_colors"
    _set_default_colors_categorical(adata, feature)
    adata.uns[ckey] = [color_mapping.get(c, adata.uns[ckey][i]) for i, c in enumerate(cats)]


def _set_default_colors_categorical(
    adata: AnnData,
    feature: str,
):
    """Populate default category colors when Scanpy-style colors are missing."""
    color_key = feature + "_colors"
    if color_key not in adata.uns:
        cats = adata.obs[feature].cat.categories
        if len(cats) <= 20:
            adata.uns[color_key] = default_20[:len(cats)]
        elif len(cats) <= 28:
            adata.uns[color_key] = default_28[:len(cats)]
        elif len(cats) <= 102:
            adata.uns[color_key] = default_102[:len(cats)]
        else:
            raise ValueError("Categorical column has more than 102 categories!")


def get_categorical_colormap(
    adata: AnnData,
    feature: str,
) -> dict:
    """Return the current category-to-color mapping for a categorical feature."""
    color_key = feature + "_colors"
    if color_key not in adata.uns:
        raise ValueError(f"Color mapping for '{feature}' not present in adata.uns")
    return dict(zip(adata.obs[feature].cat.categories, adata.uns[color_key]))


def subplots_with_side_axis(
    fig: Figure,
    nrows: int,
    ncols: int,
    side_ax_direction: Literal["horizontal", "vertical"],
    side_ax_proportion: float,
):
    """Create a subplot grid with one extra shared axis for a legend or colorbar.

    Parameters
    ----------
    fig
        Figure that should receive the axes.
    nrows, ncols
        Grid shape for the main plotting axes.
    side_ax_direction
        Whether the extra axis should be appended below the grid or to its right.
    side_ax_proportion
        Fraction of the figure reserved for the extra axis.

    Returns
    -------
    tuple[list[Axes], Axes]
        The main axes in row-major order and the shared side axis.
    """
    # Prepare figure with 2x2 plots and one extra (for colorbar/legend)
    if side_ax_direction == "horizontal":
        gs = GridSpec(nrows+1, ncols, height_ratios=[(1 - side_ax_proportion)/nrows]*nrows + [side_ax_proportion])
        axes = [fig.add_subplot(gs[i // ncols, i % ncols]) for i in range(nrows * ncols)]
        side_ax = fig.add_subplot(gs[nrows, :])  # bottommost row
    elif side_ax_direction == "vertical":
        gs = GridSpec(nrows, ncols+1, width_ratios=[(1 - side_ax_proportion)/ncols]*ncols + [side_ax_proportion])
        axes = [fig.add_subplot(gs[i // ncols, i % ncols]) for i in range(nrows * ncols)]
        side_ax = fig.add_subplot(gs[:, ncols])  # bottommost row
    else:
        raise ValueError(f"Invalid value {side_ax_direction} for legend_loc")
    
    return axes, side_ax
