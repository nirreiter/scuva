"""Composition-style plots for summarizing categorical observations."""
from __future__ import annotations

from typing import Any, Literal

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patheffects import withStroke
import numpy as np
from pandas import DataFrame, crosstab
from anndata import AnnData

from . import DPI, _set_default_colors_categorical, _get_default_colors, get_categorical_colormap, subplots_with_legend_axis
from .legend import make_legend
from ..text import clean_title, rename
from ..util import sort_categories_handle_ints, _require_categorical

def graph_proportions(
    adata: AnnData, 
    x: str,
    y: str,
    x_order: Literal["sort"] | list | np.ndarray | None = "sort",
    figsize: tuple[int, int] = (3, 6),
    legend_proportion: float = 0.1,
    x_tick_rotation: int = 0,
    percentages_fontsize: int = 8,
    percentages_color: str = "#000000",
    percentages_outline_width: float = 0.5,
    percentages_outline_color: str = "#dddddd",
    percentages_decimal_places: int = 2,
    percentages_display_threshold: float = 1,
    combine_small_percentages_other: bool = False,
    ignore_original_colors: bool = False,
    color_override: dict[str, str] | None = None,
    legend_kwargs: dict[str, Any] | None = None,
    spines: set[str] = {"left"}
) -> tuple[DataFrame, Figure, Axes, Axes]:
    """Plot stacked percentages for one observation column across another.

    Parameters
    ----------
    adata
        AnnData object containing both observation columns.
    x
        Observation column used for the bar positions.
    y
        Observation column used for the stacked segments and legend entries.
    x_order
        Optional explicit order to apply before plotting. The default value
        ``"sort"`` sorts integer-like categories numerically and other categories in alphabetic order.
    figsize
        Figure size passed to Matplotlib.
    legend_proportion
        Fraction of the figure width reserved for the side legend axis.
    x_tick_rotation
        Rotation angle for x-axis tick labels.
    percentages_fontsize, percentages_color
        Styling for the percentage labels drawn inside each bar segment.
    percentages_outline_width, percentages_outline_color
        Stroke styling that helps the percentage labels remain readable.
    percentages_decimal_places
        Number of decimal places to show. Only 0, 1, or 2 are accepted.
    percentages_display_threshold
        Segments at or below this percentage are left unlabeled.
    combine_small_percentages_other
        If ``True``, collapse categories whose maximum proportion stays strictly
        below ``percentages_display_threshold`` into a combined ``Other`` column.
        Will fail if a column named ``Other`` already exists.
    ignore_original_colors
        If ``True``, rebuild colors from the default Scanpy palettes instead of
        using ``adata.uns`` colors.
    color_override
        Optional per-category color mapping that overrides stored colors for ``y``.
    legend_kwargs
        Extra keyword arguments forwarded to :func:`make_legend` after the helper's
        default title and renaming settings are applied.
    spines
        Subset of axis spines that should remain visible.

    Returns
    -------
    tuple[DataFrame, Figure, Axes, Axes]
        The percentage table used for plotting and the created figure axes.
    """
    if percentages_decimal_places < 0 or percentages_decimal_places > 2:
        raise ValueError("Can only display 0, 1, or 2 decimal places for percentages")
    _require_categorical(adata, x)
    _require_categorical(adata, y)
    
    df = crosstab(adata.obs[x], adata.obs[y])
    
    if x_order == "sort":
        x_order = sort_categories_handle_ints(adata.obs[x].cat.categories)
    if x_order is not None:
        df = df.loc[x_order, :]
    
    fig = plt.figure(figsize=figsize, dpi=DPI)
    axes, side_ax = subplots_with_legend_axis(fig, 1, 1, 1, "vertical", legend_proportion)
    ax = axes[0]
    # make percent
    df: DataFrame = df.div(df.sum(axis=1), axis=0) * 100

    # combine columns whose proportion never crosses the threshold into 'Other'
    small_cols = None
    if percentages_display_threshold > 0 and combine_small_percentages_other:
        if "Other" in df.columns:
            raise ValueError("'Other' is already a value and can't be used to combine small percentages!")
        column_max = df.max(axis=0)
        small_cols = column_max[column_max < percentages_display_threshold].index.tolist()
        if len(small_cols) > 0:
            other_col = df[small_cols].sum(axis=1)
            df = df.drop(columns=small_cols)
            df['Other'] = other_col
    
    if ignore_original_colors:
        if combine_small_percentages_other and "Other" in df.columns:
            cats = df.columns.drop('Other')
            colormap = dict(zip(cats, _get_default_colors(len(cats))))
            colormap['Other'] = '#cccccc'
        else:
            cats = df.columns
            colormap = dict(zip(cats, _get_default_colors(len(cats))))
    else:
        _set_default_colors_categorical(adata, y)
        colormap = get_categorical_colormap(adata, y)
        # ensure color for combined 'Other' if present
        if 'Other' in df.columns and 'Other' not in colormap:
            colormap['Other'] = '#cccccc'
    
    if color_override is not None:
        colormap |= color_override
    
    if small_cols is not None:
        for col in small_cols:
            if col in colormap:
                del colormap[col]
    
    bottom = np.zeros(len(df.index))
    x_positions = range(len(df.index))
    for ct in df.columns:
        ax.bar(
            x_positions,
            df[ct],
            bottom = bottom,
            label = ct,
            color = colormap[ct],
        )

        # place text in the middle of the bar segments showing the value
        if percentages_fontsize > 0 and percentages_display_threshold < 100:
            for i, i_x in enumerate(df.index):
                height = float(df.loc[i_x, ct]) # pyright: ignore[reportArgumentType]
                if height <= percentages_display_threshold:
                    continue
                match percentages_decimal_places:
                    case 0: text = f"{height:.0f}%"
                    case 1: text = f"{height:.1f}%"
                    case 2: text = f"{height:.2f}%"
                ax.text(
                    i, 
                    bottom[i] + height/2,
                    text,
                    ha="center", va="center",
                    fontsize=percentages_fontsize, 
                    color=percentages_color, 
                    weight="bold",
                    path_effects=[withStroke(
                        linewidth=percentages_outline_width, 
                        foreground=percentages_outline_color
                    )]
                )
        
        bottom += df[ct]
    
    ax.set_ylim((0, 100))
    ax.grid(False)
    ax.set_ylabel("Percent of Cells")
    ax.set_xlabel("")
    ax.set_xticks(
        range(len(df.index)),
        [clean_title(rename(adata, l)) for l in df.index],
        rotation=x_tick_rotation,
    )
    for spine in ["top", "right", "bottom", "left"]:
        ax.spines[spine].set_visible(spine in spines)
    if "bottom" not in spines:
        ax.tick_params(length=0, axis="x")
    
    _legend_kwargs: dict[str, Any] = dict(
        title=clean_title(rename(adata, y)),
        label_rename_dict={l: clean_title(rename(adata, l)) for l in colormap.keys()},
        sort_ints="no",
    )
    if legend_kwargs is not None:
        _legend_kwargs |= legend_kwargs
    
    make_legend(
        ax=side_ax,
        label_color_dict=dict(reversed(colormap.items())),
        **_legend_kwargs
    )
    plt.tight_layout()
    return df, fig, ax, side_ax

def graph_counts(
    adata: AnnData, 
    hue: str, 
    x: str,
    x_order: list | np.ndarray | None = None,
    stack: bool = False,
    sort_by_size: bool = True,
    figsize=(6, 3),
    legend_proportion: float = 0.1,
    x_tick_rotation: int = 90,
    legend_kwargs: dict[str, Any] | None = None
) -> tuple[DataFrame, Figure, Axes, Axes]:
    """Plot category counts as grouped or stacked bars.

    Parameters
    ----------
    adata
        AnnData object containing the observation columns.
    hue
        Observation column used for bar colors and legend entries. This
        column is expected to be categorical.
    x
        Observation column used for the x-axis groups.
    x_order
        Optional explicit order to apply before plotting when size-based sorting is
        disabled. Ignored when ``sort_by_size=True``.
    stack
        If ``True``, stack the category bars instead of placing them side by side.
    sort_by_size
        If ``True``, order groups by total count before plotting.
    figsize
        Figure size passed to Matplotlib.
    legend_proportion
        Fraction of the figure width reserved for the side legend axis.
    x_tick_rotation
        Rotation angle for x-axis tick labels.
    legend_kwargs
        Extra keyword arguments forwarded to :func:`make_legend`.

    Returns
    -------
    tuple[DataFrame, Figure, Axes, Axes]
        The count table used for plotting and the created figure axes.
    """
    if x not in adata.obs.columns:
        raise ValueError(f"'{x}' not in adata.obs")
    if hue not in adata.obs.columns:
        raise ValueError(f"'{hue}' not in adata.obs")
    df = crosstab(adata.obs[x], adata.obs[hue])
    if sort_by_size:
        df = df.assign(total=lambda c: c.sum(axis=1))
        df = df.sort_values("total", ascending=False)
        df = df.drop(columns="total")
    elif x_order is not None:
        df = df.loc[x_order, :]
    
    _set_default_colors_categorical(adata, hue)
    colormap = get_categorical_colormap(adata, hue)
    fig = plt.figure(figsize=figsize)
    axes, side_ax = subplots_with_legend_axis(fig, 1, 1, 1, "vertical", legend_proportion)
    ax = axes[0]
    df.plot.bar(figsize=figsize, stacked=stack, color=colormap, ax=ax, legend=False)

    fig.set_dpi(DPI)
    ax.set_axisbelow(True)
    ax.grid(axis="y")
    ax.set_ylabel("Number of Cells")
    ax.set_xlabel(clean_title(rename(adata, x)))
    ax.set_xticks(
        range(len(df.index)), 
        [clean_title(rename(adata, l)) for l in df.index], 
        rotation=x_tick_rotation,
    )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if legend_kwargs is None:
        legend_kwargs = dict()
    make_legend(
        ax=side_ax,
        label_color_dict=colormap,
        title=clean_title(rename(adata, hue)),
        label_rename_dict={l: clean_title(rename(adata, l)) for l in colormap.keys()},
        **legend_kwargs
    )
    
    plt.tight_layout()
    return df, fig, ax, side_ax
