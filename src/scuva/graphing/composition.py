"""Composition-style plots for summarizing categorical observations."""

from typing import Any

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patheffects import withStroke
import numpy as np
from pandas import DataFrame, crosstab
from anndata import AnnData

from . import DPI, _set_default_colors_categorical, get_categorical_colormap, subplots_with_side_axis
from .legend import make_legend
from ..text import clean_title, rename

def graph_proportions(
    adata: AnnData, 
    x: str,
    y: str,
    x_order: list | np.ndarray | None = None,
    figsize: tuple[int, int] = (2, 4),
    legend_proportion: float = 0.1,
    x_tick_rotation: int = 0,
    percentages_fontsize: int = 8,
    percentages_color: str = "#0000000",
    percentages_outline_width: float = 0.5,
    percentages_outline_color: str = "#dddddd",
    percentages_decimal_places: int = 2,
    percentages_display_threshold: float = 1,
    color_override: dict[str, str] | None = None,
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
        Optional explicit order to apply before plotting.
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
    color_override
        Optional per-category color mapping that overrides stored colors for ``y``.

    Returns
    -------
    tuple[DataFrame, Figure, Axes, Axes]
        The percentage table and the created figure axes.
    """
    if percentages_decimal_places < 0 or percentages_decimal_places > 2:
        raise ValueError("Can only display 0, 1, or 2 decimal places for percentages")
    if x not in adata.obs.columns:
        raise ValueError(f"'{x}' not in adata.obs")
    if y not in adata.obs.columns:
        raise ValueError(f"'{y}' not in adata.obs")
    df = crosstab(adata.obs[x], adata.obs[y])
    if x_order is not None:
        df = df.loc[:, x_order]
    
    fig = plt.figure(figsize=figsize, dpi=DPI)
    axes, side_ax = subplots_with_side_axis(fig, 1, 1, "vertical", legend_proportion)
    ax = axes[0]
    # make percent
    df = df.div(df.sum(axis=1), axis=0) * 100
    
    bottom = np.zeros(len(df.index))
    
    _set_default_colors_categorical(adata, y)
    colormap = get_categorical_colormap(adata, y)
    for ct in df.columns:
        ax.bar(
            df.index,
            df[ct],
            bottom = bottom,
            label = ct,
            color = color_override[ct] if color_override is not None else colormap[ct],
        )

        # place text in the middle of the bar segments showing the value
        for i, i_x in enumerate(df.index):
            height = float(df.loc[i_x, ct]) # pyright: ignore[reportArgumentType]
            if height <= percentages_display_threshold:
                continue
            match percentages_decimal_places:
                case 0: text = f"{height:.0f}%"
                case 1: text = f"{height:.1f}%"
                case 2: text = f"{height:.2f}%"
            ax.text(
                i_x, 
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
    
    ax.set_xlim((-0.5, 1.5))
    ax.set_ylim((0, 100))
    ax.grid(False)
    ax.set_ylabel("Percent of Cells")
    ax.set_xlabel("")
    locs, labels = plt.xticks()
    for l in labels:
        l.set_text(clean_title(rename(adata, l.get_text())))
    ax.set_xticks(locs, labels, rotation=x_tick_rotation)
    ax.tick_params(length=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    make_legend(
        ax=side_ax,
        label_color_dict=colormap,
        title=clean_title(rename(adata, y)),
        label_rename_dict={l: clean_title(rename(adata, l)) for l in colormap.keys()}
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
    figsize=(8, 4),
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
        Observation column used for bar colors and legend entries.
    x
        Observation column used for the x-axis groups.
    x_order
        Optional explicit order to apply before plotting when size-based sorting is
        disabled.
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
        The count table and the created figure axes.
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
        df = df.loc[:, x_order]
    
    _set_default_colors_categorical(adata, hue)
    colormap = get_categorical_colormap(adata, hue)
    fig = plt.figure(figsize=figsize)
    axes, side_ax = subplots_with_side_axis(fig, 1, 1, "vertical", legend_proportion)
    ax = axes[0]
    df.plot.bar(figsize=figsize, stacked=stack, color=colormap, ax=ax, legend=False)
    
    fig = ax.get_figure()
    if fig is not None:
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
