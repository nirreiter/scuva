"""UMAP plotting helpers for categorical and continuous single-cell features."""
from __future__ import annotations

from matplotlib import pyplot as plt
from matplotlib import colors as mplc
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec

import numpy as np
import numpy.typing as npt
from anndata import AnnData
from pandas import CategoricalDtype
from scanpy.get import obs_df

from typing import Literal, Any

from . import DEFAULT_CMAP, DPI, _set_default_colors_categorical, get_categorical_colormap, subplots_with_legend_axis, _adjust_axis_for_legend
from ..text import clean_title, rename
from .legend import make_colorbar, make_legend
from ..util import is_numeric


POINT_SIZE_FACTOR = 1000


def _minmax_int_slow_with_zero(data):
    """Return integer plot bounds spanning the data while forcing zero inside them."""
    return int(np.floor(min(min(data), 0))), int(np.ceil(max(max(data), 0)))


def _select_point_size(
    umap: np.ndarray, 
    xlim: tuple[float, float] | None,
    ylim: tuple[float, float] | None
):
    """Estimate a scatter marker size from the visible embedding extent.

    The heuristic uses the visible width and height of the embedding so denser
    layouts receive smaller points.
    """
    if xlim is not None:
        x = xlim[1] - xlim[0]
    else:
        umap_x = umap[:, 0]
        x = max(umap_x) - min(umap_x)
    
    if ylim is not None:
        y = ylim[1] - ylim[0]
    else:
        umap_y = umap[:, 1]
        y = max(umap_y) - min(umap_y)
    
    return POINT_SIZE_FACTOR / (x * y)


def _normalize_bottom_points(
    bottom_points: npt.NDArray[np.intp] | npt.NDArray[np.bool_] | None,
    size: int,
) -> npt.NDArray[np.bool_] | None:
    """Normalize bottom-point selections into a boolean mask.

    Boolean masks are validated and copied. Integer index arrays are converted into
    a mask of length ``size``.
    """
    if bottom_points is None:
        return None

    points = np.asarray(bottom_points)
    if points.dtype == bool:
        if len(points) != size:
            raise ValueError("Boolean bottom_points mask must match the number of observations.")
        return points.astype(bool, copy=True)

    indices = points.astype(np.intp, copy=False)
    if np.any((indices < 0) | (indices >= size)):
        raise ValueError("bottom_points indices must be within the plotted observation range.")

    mask = np.zeros(size, dtype=bool)
    mask[indices] = True
    return mask


def _clear_axis(ax: Axes) -> None:
    """Strip an axis down to an invisible placeholder panel."""
    ax.set_facecolor("none")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.patch.set_alpha(0)


def _make_umap_legend(
    ax: Axes,
    adata: AnnData, 
    feature: str, 
    color_dict: dict | None = None,
    legend_order: list | np.ndarray | None = None, 
    legend_renaming: dict[str, str] | None = None,
    **legend_kwargs
):
    """Create a categorical legend for a UMAP panel using AnnData metadata.

    Category order is taken from ``legend_order`` when provided, otherwise from the
    categorical order stored on ``adata.obs[feature].cat.categories``. Display labels are passed
    through :func:`rename` before the legend is drawn.
    """
    categories = list(adata.obs[feature].cat.categories)
    if color_dict is None:
        color_dict = get_categorical_colormap(adata, feature)

    if legend_order is not None:
        ordered_categories = [category for category in legend_order if category in color_dict]
    else:
        ordered_categories = [category for category in categories if category in color_dict]

    display_color_dict = {
        rename(adata, str(category), legend_renaming): color_dict[category]
        for category in ordered_categories
    }
    
    if "title" in legend_kwargs:
        title = legend_kwargs["title"]
        del legend_kwargs["title"]
    else:
        title = clean_title(rename(adata, feature))
    
    make_legend(ax, title, display_color_dict, **legend_kwargs)

def umap(
    adata: AnnData, 
    feature: str, 
    use_raw: bool = False,
    layer: str | None = None,
    cmap: mplc.Colormap | dict = DEFAULT_CMAP,
    umap_obsm_key: str = "X_umap",
    figsize : tuple[int, int] = (4, 4),
    legend_kwargs: dict[str, Any] | None = None,
    legend_order: list | np.ndarray | None = None,
    legend_loc: str | None = "on data",
    legend_renaming: dict[str, str] | None = None,
    ax: Axes | None = None, 
    side_ax: Axes | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    vcenter: float | None = None,
    s: float | None = None,
    a: float | None = None,
    show_grid: bool = False,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    bottom_points: npt.NDArray[np.intp] | npt.NDArray[np.bool_] | None = None,
    sort_by_abs: bool = True,
    **kwargs: Any,
):
    """Plot a UMAP embedding colored by a categorical or continuous feature.

    Parameters
    ----------
    adata
        AnnData object containing the embedding and feature values.
    feature
        Observation column or expression feature to visualize.
    use_raw
        Read expression values from ``adata.raw`` instead of the main matrix.
    layer
        Layer name to read expression values from. Mutually exclusive with
        ``use_raw``.
    cmap
        Colormap for continuous data, or a category-to-color mapping for
        categorical data.
    umap_obsm_key
        Key in ``adata.obsm`` containing the 2D embedding.
    figsize
        Figure size used when creating new axes.
    legend_kwargs
        Extra keyword arguments forwarded to :func:`make_colorbar` for continuous
        features, to :func:`make_legend` for side legends, or to ``Axes.text`` for
        on-data categorical labels.
    legend_order
        Optional category order for side legends.
    legend_loc
        Location strategy for categorical legends. For continuous plots on newly
        created axes, ``"outside right"`` also enables the side axis that holds
        the colorbar.
        'outside right' places a legend outside the graph. 
        'on data' places text on top of the data, intended for cluster-like data.
        Any other string places the legend on the same axis as the data.
        If this argument is set to `None`, no legend is drawn.
    legend_renaming
        Optional label overrides used for legend and title text.
    ax, side_ax
        Existing axes for the scatter plot and its legend or colorbar. If ``ax`` is
        omitted, new axes are created.
    vmin, vmax, vcenter
        Color scaling parameters for continuous data.
    s
        Scatter marker size. Defaults to a size derived from the embedding extent.
    a
        Scatter alpha value.
    show_grid
        Draw integer grid lines spanning the UMAP extent.
    xlim, ylim
        Optional axis bounds.
    bottom_points
        Boolean mask or integer index array identifying points that should be drawn
        first beneath the rest.
    sort_by_abs
        For continuous features only, draw points in order of increasing absolute
        value so larger magnitudes appear on top. When ``False``, points are
        shuffled instead. For categorical data, points are always shuffled.
    **kwargs
        Additional keyword arguments forwarded to every ``Axes.scatter`` call after
        adding ``edgecolors='none'`` by default.

    Returns
    -------
    tuple[Axes, Axes | None, list[Any]]
        The main axes, the side axes if one was used, and any on-data text labels
        added for categorical plots.

    Raises
    ------
    ValueError
        If the requested data source or UMAP embedding is unavailable, or if the
        colormap configuration does not match the feature type.

    Notes
    -----
    Continuous plots first draw every point in light gray, then overplot only the
    nonzero values in color. For categorical plots, a non-dict ``cmap`` argument is
    ignored and colors come from ``adata.uns`` or Scanpy defaults.
    """
    #* Argument validation
    if layer is not None and use_raw:
        raise ValueError("Either select use_raw=True or a layer name, not both.")
    if use_raw and adata.raw is None:
        raise ValueError("Cannot select use_raw=True if no raw data is available in the anndata object.")
    if layer is not None and layer not in adata.layers:
        raise ValueError(f"Layer '{layer}' is not present in the anndata object.")
    
    is_categorical = feature in adata.obs and isinstance(adata.obs[feature].dtype, CategoricalDtype)
    if not is_categorical and isinstance(cmap, dict):
        raise ValueError("Colormap can only be a dictionary for categorical data.")
    
    if umap_obsm_key is None:
        if "X_umap" not in adata.obsm:
            raise ValueError(
                "UMAP data is not present in your anndata object at the default location of adata.obsm['X_umap']."
                "Please generate the UMAP first or provide the obsm key with the 'umap_obsm_key' parameter."
            )
    else:
        if umap_obsm_key not in adata.obsm:
            raise ValueError(
                f"UMAP data is not present in your anndata object at adata.obsm['{umap_obsm_key}']."
                "Please generate the UMAP first or provide the obsm key with the 'umap_obsm_key' parameter."
            )
    
    values = obs_df(adata, keys=[feature], use_raw = use_raw, layer = layer)[feature].to_numpy()
    if len(values) == 0:
        raise ValueError("Feature exists but length of values is zero. There may be no observations in the anndata object.")
    if not is_categorical and not is_numeric(values):
        raise ValueError("Columns must be either categorical or numeric for graphing. If you intended to graph a categorical variable, cast the column to categorical like so: "
                         + f"\nadata.obs[{feature}] = adata.obs[{feature}].astype('category')\n"
                         + "If you inteded to graph a numeric column, check its values to ensure they are numeric.")
    
    kwargs = dict(
        edgecolors = "none"
    ) | kwargs
    
    #* Create the graph
    if ax is None:
        fig = plt.figure(figsize = figsize, dpi=DPI)
        gs = GridSpec(1, 2, width_ratios=[1, 0.05])
        ax = fig.add_subplot(gs[0, 0])
        if side_ax is None:
            if legend_loc == "outside right":
                side_ax = fig.add_subplot(gs[0, 1])
    else:
        gs = None
    
    if is_categorical and legend_loc == "outside right" and not side_ax:
        raise ValueError("Legend is set outside but there is no side axis provided")
    
    #* Get UMAP graphing parameters
    X_umap = np.asarray(adata.obsm[umap_obsm_key or "X_umap"])
    umap_x = X_umap[:, 0]
    umap_y = X_umap[:, 1]
    bottom_mask = _normalize_bottom_points(bottom_points, len(X_umap))
    
    s = s if s is not None else _select_point_size(X_umap, xlim, ylim)
    
    groups = None
    added_text = []
    
    #* Categorical data
    if is_categorical:
        
        ##* get colors for each point
        groups = adata.obs[feature].cat.categories
        if isinstance(cmap, dict):
            missing_categories = [category for category in groups if category not in cmap]
            if missing_categories:
                raise ValueError(
                    "Categorical colormap is missing colors for: "
                    + ", ".join(str(category) for category in missing_categories)
                )
            colors = np.asarray([cmap[category] for category in groups], dtype=object)
            point_colors = adata.obs[feature].astype(object).map(cmap).to_numpy()
        else:
            _set_default_colors_categorical(adata, feature)
            colors = np.asarray(adata.uns[feature + "_colors"]) # if (feature + "_colors") in adata.uns else sns.color_palette(n_colors=len(groups))
            point_codes = adata.obs[feature].cat.codes.to_numpy()
            point_colors = colors[point_codes]
        
        ##* shuffle points to prevent 1 group from appearing on top
        shuffle_index = np.arange(len(umap_x))
        np.random.shuffle(shuffle_index)
        umap_x = umap_x[shuffle_index]
        umap_y = umap_y[shuffle_index]
        point_colors = point_colors[shuffle_index]
        if bottom_mask is not None:
            bottom_mask = bottom_mask[shuffle_index]
        
        ##* Graph bottom points if provided
        if bottom_mask is not None:
            ax.scatter(
                umap_x[bottom_mask],
                umap_y[bottom_mask],
                c=point_colors[bottom_mask],
                s=s,
                alpha=a,
                **kwargs,
            )
            
            umap_x = umap_x[~bottom_mask]
            umap_y = umap_y[~bottom_mask]
            point_colors = point_colors[~bottom_mask]
        
        ##* Graph points
        ax.scatter(
            umap_x,
            umap_y,
            c=point_colors,
            s=s,
            alpha=a,
            **kwargs,
        )
        
        ##* Legend directly on data
        if legend_loc == "on data":
            # Place label at median position of each group
            # Use the original (unshuffled) embedding coordinates so masks align
            orig_coords = np.asarray(adata.obsm[umap_obsm_key or "X_umap"])
            for idx, cat in enumerate(groups):
                _bbox = dict(facecolor="white", edgecolor=colors[idx], boxstyle="round,pad=0.2", alpha=0.7)
                _legend_kwargs: dict[str, Any] = dict(
                    fontsize=10, 
                    weight="bold", 
                    color="black", 
                    ha="center", 
                    va="center",
                    bbox=_bbox
                )
                # replace parameters in legend_kwargs and do smart replace for bbox
                if legend_kwargs is not None:
                    _legend_kwargs |= legend_kwargs
                    if "bbox" in legend_kwargs:
                        _legend_kwargs["bbox"] = _bbox | legend_kwargs["bbox"]
                    
                mask = (adata.obs[feature] == cat).to_numpy()
                median_x = float(np.median(orig_coords[mask, 0]))
                median_y = float(np.median(orig_coords[mask, 1]))
                added_text.append(ax.text(
                    median_x,
                    median_y,
                    clean_title(rename(adata, str(cat), legend_renaming)),
                    **_legend_kwargs
                ))
            if side_ax:
                side_ax.axis("off")
        
        ##* Make a legend
        elif legend_loc is not None:
            _legend_kwargs: dict[str, Any] = dict(
                fontsize = 10,
                hide_borders = legend_loc == "outside right"
            )
            if legend_kwargs is not None:
                _legend_kwargs |= legend_kwargs
            color_dict = dict(zip(groups, colors))
            _make_umap_legend(
                side_ax if legend_loc == "outside right" else ax, 
                adata, 
                feature, 
                color_dict=color_dict,
                legend_order=legend_order, 
                legend_renaming={k: clean_title(rename(adata, k, legend_renaming)) for k in groups},
                loc="center left" if legend_loc == "outside right" else legend_loc,
                **(_legend_kwargs or {}),
            )
            
    #* Continuous data
    else:
        if vmin is None:
            vmin = np.nanmin(values)
        if vmax is None:
            vmax = np.nanmax(values)
        
        ##* Place light grey points w/o transparency to indicate each point (prevents 'invisible' points)
        ax.scatter(
            umap_x,
            umap_y,
            c="lightgrey",
            s=s,
            alpha=1,
            **kwargs,
        )
        
        nonzero = values != 0
        umap_x = umap_x[nonzero]
        umap_y = umap_y[nonzero]
        values = values[nonzero]
        if bottom_mask is not None:
            bottom_mask = bottom_mask[nonzero]
        
        ##* Setup shared ScalarMappable for colorbar and graph
        if vcenter is not None:
            norm = mplc.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        else:
            norm = mplc.Normalize(vmin=vmin, vmax=vmax)
        assert isinstance(cmap, mplc.Colormap) # guaranteed by argument validation above
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        point_colors = sm.to_rgba(values)
        
        ##* if requested, sort by absolute value so more extreme values are visible on top
        if (sort_by_abs):
            sort_idx = np.argsort(np.abs(values))
            umap_x = umap_x[sort_idx]
            umap_y = umap_y[sort_idx]
            point_colors = point_colors[sort_idx]
            if bottom_mask is not None:
                bottom_mask = bottom_mask[sort_idx]
        
        ##* otherwise shuffle points to prevent any bias in values appearing on top
        else:
            shuffle_index = np.arange(len(umap_x))
            np.random.shuffle(shuffle_index)
            umap_x = umap_x[shuffle_index]
            umap_y = umap_y[shuffle_index]
            point_colors = point_colors[shuffle_index]
            if bottom_mask is not None:
                bottom_mask = bottom_mask[shuffle_index]
        
        ##* Graph bottom points if provided
        if bottom_mask is not None:
            ax.scatter(
                umap_x[bottom_mask],
                umap_y[bottom_mask],
                c=point_colors[bottom_mask],
                s=s,
                alpha=a,
                **kwargs,
            )
            
            umap_x = umap_x[~bottom_mask]
            umap_y = umap_y[~bottom_mask]
            point_colors = point_colors[~bottom_mask]
    
        ##* Graph points
        ax.scatter(
            umap_x,
            umap_y,
            c=point_colors,
            s=s,
            alpha=a,
            **kwargs,
        )
        
        ##* create a colorbar if a side axis is provided
        if side_ax:
            make_colorbar(
                sm = sm, 
                cax = side_ax, 
                label = f"{rename(adata, feature)} Expression", 
                ticks = None, #! TODO: Support for ticks
                vmin = vmin,
                vmax = vmax,
                vcenter = vcenter,
                **(legend_kwargs or {}),
            )
    
    #* Final graph styling

    ax.set_title(clean_title(rename(adata, feature)))
    ax.set_xlabel(rename(adata, "UMAP 1"))
    ax.set_ylabel(rename(adata, "UMAP 2"))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.grid(show_grid)
    if show_grid:
        for x in range(*_minmax_int_slow_with_zero(adata.obsm[umap_obsm_key or "X_umap"][:, 0])):
            ax.axvline(x=x, color="grey")
        for y in range(*_minmax_int_slow_with_zero(adata.obsm[umap_obsm_key or "X_umap"][:, 1])):
            ax.axhline(y=y, color="grey")
        ax.axvline(x=0, color="black")
        ax.axhline(y=0, color="black")
    
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    
    if gs is not None:
        plt.tight_layout()
    return ax, side_ax, added_text

def multiple_umap(
    adata: AnnData | list[AnnData], 
    features: list[str], 
    use_raw: bool = False,
    layer: str | None = None,
    cmap: mplc.Colormap | dict = DEFAULT_CMAP,
    umap_obsm_key: str = "X_umap",
    individual_figsize: tuple[int, int] = (4, 4),
    ncols: int = 2,
    legend_kwargs: dict[str, Any] | None = None,
    legend_order: list | np.ndarray | None = None,
    legend_loc: str | None | list[str | None] = "on data",
    legend_renaming: dict[str, str] | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    vcenter: float | None = None,
    s: float | None = None,
    a: float | None = None,
    show_grid: bool = False,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    sort_by_abs: bool = True,
    **kwargs: Any,
):
    """Create a grid of UMAP plots across one or more datasets and features.

    Parameters
    ----------
    adata
        A single AnnData object or a list of AnnData objects to plot.
    features
        Features to visualize for each dataset.
    use_raw, layer
        Passed through to :func:`umap` for every panel.
    cmap
        Colormap or categorical color mapping passed through to :func:`umap`.
    umap_obsm_key
        Key in ``adata.obsm`` containing the embedding coordinates.
    individual_figsize
        Base figure size allocated per plotted panel before the full grid is
        assembled.
    ncols
        Requested number of plot panels per row. The current layout implementation
        supports at most two panels per row.
    legend_kwargs, legend_order, legend_renaming
        Legend configuration forwarded to :func:`umap` for each panel.
    legend_loc
        Legend placement strategy forwarded to :func:`umap`. A list value is cycled
        across panels.
    vmin, vmax, vcenter, s, a, show_grid, xlim, ylim, sort_by_abs
        Plotting options forwarded to :func:`umap` for each panel.
    **kwargs
        Additional keyword arguments forwarded to :func:`umap`.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing all generated panels.

    Notes
    -----
    This helper always passes ``bottom_points=None`` to the underlying
    :func:`umap` calls. The current grid construction assumes one or two plot panels
    per row.
    """
    if isinstance(adata, AnnData):
        adata = [adata]
    if not isinstance(legend_loc, list):
        legend_loc = [legend_loc]
    if ncols != 1 and ncols != 2:
        raise ValueError("`ncols` supports only 1 or 2")
    
    ngraphs = len(adata) * len(features)
    w = min(ngraphs, ncols)
    h = int(np.ceil(ngraphs / w))
    fig, axes = plt.subplots(
        h, 2 if w == 1 else 5, figsize=(
            individual_figsize[0] * w + 2 * (w - 1), 
            individual_figsize[1] * h + 1 * (h - 1)
        ), 
        gridspec_kw = {
            'width_ratios': [10, 1] if w == 1 else [10, 1, 1, 10, 1],
        },
        dpi = DPI,
    )
    if h == 1:
        axes = axes.reshape(1, -1)
    if w == 1 and h == 1:
        axes = axes.reshape(1, 2)
    
    added_text = []
    
    for y in range(h):
        for x in range(w):
            index = y * w + x
            ax: Axes = axes[y, x*3]
            side_ax: Axes = axes[y, x*3 + 1]
            if index < ngraphs:
                _legend_loc = legend_loc[index % len(legend_loc)]
                if _legend_loc != "outside right":
                    _clear_axis(side_ax)
                added_text.append(umap(
                    adata = adata[index // len(features)],
                    feature = features[index % len(features)], 
                    use_raw = use_raw,
                    layer = layer,
                    cmap = cmap,
                    umap_obsm_key = umap_obsm_key,
                    legend_kwargs = legend_kwargs,
                    legend_order = legend_order,
                    legend_loc = _legend_loc,
                    legend_renaming = legend_renaming, 
                    ax = ax, 
                    side_ax = side_ax,
                    vmin = vmin,
                    vmax = vmax,
                    vcenter = vcenter,
                    s = s,
                    a = a,
                    show_grid = show_grid,
                    xlim = xlim,
                    ylim = ylim,
                    bottom_points = None, # Not supported
                    sort_by_abs = sort_by_abs,
                    **kwargs
                )[2])
            else:
                _clear_axis(ax)
                _clear_axis(side_ax)
    
    if w > 1:
        for y in range(h):
            _clear_axis(axes[y, 2])
    
    plt.subplots_adjust(
        left=0.08,
        right=0.92,  
        top=0.95,
        bottom=0.05,
        hspace=0.2,
        wspace=0.1
    )
    
    return fig
            
def umap_split(
    adata: AnnData, 
    feature: str,
    group_key: str,
    umap_obsm_key: str = "X_umap",
    legend_portion: float = 0.1,
    legend_use_extra_axis: bool = True,
    legend_kws: dict[str, Any] | None = None,
    legend_orientation: Literal["horizontal", "vertical"] | None = "horizontal",
    legend_order: list | np.ndarray | None = None,
    figsize: tuple[int, int] = (8, 8),
    cmap: mplc.Colormap = DEFAULT_CMAP,
    s: float | None = None,
    a: float | None = None,
    ncols: int = 2,
    vcenter: float | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    bottom_points: npt.NDArray[np.intp] | npt.NDArray[np.bool_] | None = None,
    **kwargs: Any
):
    """Plot a feature on separate UMAP panels for each value of a grouping column.

    Parameters
    ----------
    adata
        AnnData object containing the embedding and group assignments.
    feature
        Feature or observation column to display in each panel.
    group_key
        Observation column used to split the data into subplots.
    umap_obsm_key
        Key in ``adata.obsm`` containing the embedding coordinates.
    legend_portion
        Fraction of the layout reserved for the shared legend or colorbar axis.
    legend_use_extra_axis
        When ``True``, reuse an unused subplot cell for the shared legend or
        colorbar when one is available.
    legend_kws
        Additional keyword arguments applied to the shared legend or colorbar.
    legend_orientation
        Place the shared legend or colorbar horizontally (below the plots) or
        vertically (to the right).
        If this argument is set to `None`, no legend or colorbar is drawn.
    legend_order
        Optional explicit category ordering for categorical legends.
    figsize
        Figure size for the multi-panel layout.
    cmap
        Colormap used for continuous data or a dictionary for categorical data.
    s
        Scatter marker size. Defaults to a size derived from the embedding extent.
    a
        Scatter alpha value forwarded to each subplot.
    ncols
        Maximum number of subplot columns.
    vcenter
        Optional center value for diverging continuous color scales.
    xlim, ylim
        Optional axis bounds applied to every subplot.
    bottom_points
        Boolean mask or integer index array identifying points that should be drawn
        beneath the rest within each subgroup.
    **kwargs
        Additional keyword arguments forwarded to :func:`umap` for each panel.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the split UMAP panels and shared legend or colorbar.

    Notes
    -----
    Panels are created in the order returned by ``adata.obs[group_key].unique()``.
    """
    fig = plt.figure(figsize=figsize, dpi=DPI)  # wider to fit legend or colorbar
    
    groups = adata.obs[group_key].unique().tolist()
    # adatas = [adata[adata.obs[group_key] == t] for t in groups]
    # if isinstance(features, str):
    #     features = [features]
    # return multiple_umap(adatas, features, **kwargs)
    
    w = min(ncols, len(groups))
    h = max(1, int(np.ceil(len(groups) / w)))

    axes, legend_ax = subplots_with_legend_axis(
        fig, len(groups), h, w, legend_orientation, legend_portion, legend_use_extra_axis
    )
    legend_ax.axis("off")
    
    X_umap = np.asarray(adata.obsm[umap_obsm_key or "X_umap"])
    s = s if s is not None else _select_point_size(X_umap, xlim, ylim)
    bottom_mask = _normalize_bottom_points(bottom_points, adata.n_obs)

    is_categorical = feature in adata.obs and isinstance(adata.obs[feature].dtype, CategoricalDtype)

    if not is_categorical:
        # Compute shared vmin and vmax for continuous features
        values = obs_df(adata, keys=[feature])[feature].to_numpy()
        vmin = np.nanmin(values)
        vmax = np.nanmax(values)

        # Setup shared ScalarMappable for colorbar
        if vmin < 0 and vmax > 0:
            if vcenter is None:
                vcenter = 0
            norm = mplc.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        else:
            norm = mplc.Normalize(vmin=vmin, vmax=vmax)
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        
        kwargs = {"vmin": vmin, "vmax": vmax} | kwargs
        if vcenter is not None:
            kwargs = {"vcenter": vcenter} | kwargs

    # Create each subplot
    for i, t in enumerate(groups):
        subgroup_mask = (adata.obs[group_key] == t).to_numpy()
        adata_sub = adata[subgroup_mask]
        adata_sub.uns = adata.uns
        
        sub_bottom_points = bottom_mask[subgroup_mask] if bottom_mask is not None else None
        
        umap(
            adata=adata_sub,
            feature=feature,
            cmap=cmap,
            ax=axes[i],
            side_ax=None,
            umap_obsm_key=umap_obsm_key,
            legend_loc=None,
            s=s,
            a=a,
            xlim=xlim,
            ylim=ylim,
            bottom_points=sub_bottom_points,
            **kwargs,
        )
        axes[i].set_title(clean_title(rename(adata, str(t))))
        # axes[i].set_xlabel(None)
        # axes[i].set_ylabel(None)
    
    # clear any extra axes
    for i in range(len(groups), len(axes)):
        _clear_axis(axes[i])

    # Add either colorbar or legend
    if legend_orientation is not None:
        if is_categorical:
            legend_color_dict = None
            if isinstance(cmap, dict):
                categories = adata.obs[feature].cat.categories
                missing_categories = [category for category in categories if category not in cmap]
                if missing_categories:
                    raise ValueError(
                        "Categorical colormap is missing colors for: "
                        + ", ".join(str(category) for category in missing_categories)
                    )
                legend_color_dict = {category: cmap[category] for category in categories}
            else:
                _set_default_colors_categorical(adata, feature)
            _make_umap_legend(
                ax=legend_ax,
                adata=adata, 
                feature=feature, 
                color_dict=legend_color_dict,
                legend_order=legend_order,
                loc=("upper center" if legend_orientation == "horizontal" else "center left"),
                **(legend_kws or {}),
            )
        else:
            colorbar_title = clean_title(rename(adata, feature))
            if feature in adata.var_names:
                colorbar_title += " Expression"
            legend_ax.axis("on")
            make_colorbar(
                sm=sm,
                cax=legend_ax,
                label=colorbar_title,
                orientation=legend_orientation,
                vmin=vmin,
                vmax=vmax,
                vcenter=vcenter,
                **(legend_kws or {}),
            )
    
    plt.tight_layout()
    if w * h > len(groups) and legend_use_extra_axis:
        _adjust_axis_for_legend(legend_ax, legend_orientation, legend_portion)
    return fig
