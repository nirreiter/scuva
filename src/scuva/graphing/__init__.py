"""Shared plotting utilities for categorical colors and figure layout."""

from typing import Literal

from matplotlib import colors as mplc
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from anndata import AnnData
from scanpy.plotting.palettes import default_20, default_28, default_102

from ..util import _require_categorical

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
    _require_categorical(adata, feature)
    
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


def _get_default_colors(
    feature_count: int,
):
    """Return a Scanpy default palette sized for the requested category count.

    Up to 20, 28, or 102 categories are supported using Scanpy's bundled default
    palettes.
    """
    if feature_count <= 20:
        return default_20[:feature_count]
    elif feature_count <= 28:
        return default_28[:feature_count]
    elif feature_count <= 102:
        return default_102[:feature_count]
    else:
        raise ValueError("Categorical column has more than 102 categories!")


def _set_default_colors_categorical(
    adata: AnnData,
    feature: str,
):
    """Populate default category colors when Scanpy-style colors are missing."""
    color_key = feature + "_colors"
    if color_key not in adata.uns:
        cats = adata.obs[feature].cat.categories
        adata.uns[color_key] = _get_default_colors(feature_count=len(cats))


def get_categorical_colormap(
    adata: AnnData,
    feature: str,
) -> dict:
    """Return the current category-to-color mapping for a categorical feature.

    Unlike :func:`set_categorical_colors`, this helper does not populate missing
    colors automatically; it expects ``adata.uns[f"{feature}_colors"]`` to already
    exist.
    """
    _require_categorical(adata, feature)
    
    color_key = feature + "_colors"
    if color_key not in adata.uns:
        raise ValueError(f"Color mapping for '{feature}' not present in adata.uns")
    return dict(zip(adata.obs[feature].cat.categories, adata.uns[color_key]))


def _adjust_axis_for_legend(
    ax: Axes, 
    orientation: Literal["horizontal", "vertical"], 
    proportion: float,
):
    """Resize an existing legend or colorbar axis within its current slot.

    The adjustment keeps only a fraction of the current axis area, either by
    shrinking its height for horizontal legends or its width for vertical legends.
    """
    pos = ax.get_position()
    if orientation == "horizontal":
        ax.set_position((
            pos.x0, pos.y0 + (1 - proportion) * pos.height, 
            pos.width, proportion * pos.height
        ))
        
    elif orientation == "vertical":
        ax.set_position((
            pos.x0, pos.y0, 
            proportion * pos.width, pos.height
        ))
    else:
        raise ValueError(f"Invalid value {orientation} for side_ax_orientation")


def subplots_with_legend_axis(
    fig: Figure,
    total_subplots: int,
    nrows: int,
    ncols: int,
    side_ax_orientation: Literal["horizontal", "vertical"],
    side_ax_proportion: float,
    use_extra_subplot_axis: bool = True,
):
    """Create a subplot grid with one extra shared axis for a legend or colorbar.

    Parameters
    ----------
    fig
        Figure that should receive the axes.
    total_subplots
        Number of populated plotting axes to create inside the grid.
    nrows, ncols
        Grid shape for the main plotting axes.
    side_ax_orientation
        Whether the extra axis should be appended below the grid or to its right.
    side_ax_proportion
        Fraction of the figure reserved for the extra axis.
    use_extra_subplot_axis
        When ``True`` and the main grid already contains an unused subplot slot,
        reuse that final slot as the legend axis instead of allocating an extra row
        or column.

    Returns
    -------
    tuple[list[Axes], Axes]
        The main axes in row-major order and the shared side axis.

    Notes
    -----
    ``side_ax_orientation`` only affects layouts that allocate an extra row or
    column. If an unused subplot slot is reused, the side axis simply occupies the
    final grid cell.
    """
    # Prepare figure with wxh plots and one extra (for colorbar/legend)
    if nrows * ncols > total_subplots and use_extra_subplot_axis:
        gs = GridSpec(nrows, ncols)
        axes = [fig.add_subplot(gs[i // ncols, i % ncols]) for i in range(total_subplots)]
        legend_ax = fig.add_subplot(gs[-1, -1])
        
        return axes, legend_ax

    if side_ax_orientation == "horizontal":
        gs = GridSpec(nrows+1, ncols, height_ratios=[(1 - side_ax_proportion)/nrows]*nrows + [side_ax_proportion])
        axes = [fig.add_subplot(gs[i // ncols, i % ncols]) for i in range(total_subplots)]
        legend_ax = fig.add_subplot(gs[nrows, :])  # bottommost row
    elif side_ax_orientation == "vertical":
        gs = GridSpec(nrows, ncols+1, width_ratios=[(1 - side_ax_proportion)/ncols]*ncols + [side_ax_proportion])
        axes = [fig.add_subplot(gs[i // ncols, i % ncols]) for i in range(total_subplots)]
        legend_ax = fig.add_subplot(gs[:, ncols])  # bottommost row
    else:
        raise ValueError(f"Invalid value {side_ax_orientation} for side_ax_orientation")
    
    return axes, legend_ax
