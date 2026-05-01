"""UMAP plotting helpers for categorical and continuous single-cell features."""

from matplotlib import pyplot as plt
from matplotlib import colors as mplc
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec

import numpy as np
import numpy.typing as npt
from anndata import AnnData
from pandas import CategoricalDtype
from scanpy.get import obs_df

from typing import Literal, Any

from . import DEFAULT_CMAP, DPI, _set_default_colors_categorical, get_categorical_colormap, subplots_with_side_axis
from ..text import clean_title, rename
from .legend import make_colorbar, make_legend


POINT_SIZE_FACTOR = 1000


def _minmax_int_slow_with_zero(data):
    """Return integer plot bounds that always include zero."""
    return int(np.floor(min(min(data), 0))), int(np.ceil(max(max(data), 0)))


def _select_point_size(
    umap: np.ndarray, 
    xlim: tuple[float, float] | None,
    ylim: tuple[float, float] | None
):
    """Estimate a scatter marker size from the visible embedding extent."""
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
    """Normalize bottom-point selections into a boolean mask."""
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
    """Create a categorical legend for a UMAP panel using AnnData metadata."""
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
    figsize : tuple[int, int] = (10, 5),
    legend_kwargs: dict[str, Any] | None = None,
    legend_order: list | np.ndarray | None = None,
    legend_loc: Literal["right", "on data"] = "on data",
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
    legend_loc
        Location strategy for categorical legends. 
        'right' places a legend outside the graph. 
        'on data' places text on top of the data, intended for cluster-like data.
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
        Boolean mask identifying points that should be drawn first beneath the rest.
    legend_kwargs
        Extra keyword arguments forwarded to :func:`make_colorbar` or to :func:`ax.legend`.
    **kwargs
        Additional keyword arguments forwarded to ``Axes.scatter``.

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
    
    kwargs = dict(
        edgecolors = "none"
    ) | kwargs
    
    #* Create the graph
    if ax is None:
        fig = plt.figure(figsize = figsize, dpi=DPI)
        gs = GridSpec(1, 2, width_ratios=[1, 0.05])
        ax = fig.add_subplot(gs[0, 0])
        if side_ax is None:
            side_ax = fig.add_subplot(gs[0, 1])
    else:
        gs = None
    
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
                mask = (adata.obs[feature] == cat).to_numpy()
                median_x = float(np.median(orig_coords[mask, 0]))
                median_y = float(np.median(orig_coords[mask, 1]))
                added_text.append(ax.text(
                    median_x,
                    median_y,
                    rename(adata, str(cat), legend_renaming),
                    fontsize=10,
                    weight="bold",
                    color="black",
                    ha="center",
                    va="center",
                    bbox=dict(facecolor="white", edgecolor=colors[idx], boxstyle="round,pad=0.2", alpha=0.7)
                ))
            if side_ax:
                side_ax.axis("off")
        
        ##* Legend on the side
        elif side_ax:
            color_dict = dict(zip(groups, colors))
            _make_umap_legend(
                side_ax, 
                adata, 
                feature, 
                color_dict=color_dict,
                legend_order=legend_order, 
                legend_renaming=legend_renaming,
                loc="center left",
                **(legend_kwargs or {}),
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
    
    plt.tight_layout()
    return ax, side_ax, added_text

def multiple_umap(
    adata: AnnData | list[AnnData], 
    features: list[str], 
    cmap: mplc.Colormap | dict = DEFAULT_CMAP,
    umap_obsm: str = "X_umap",
    legend_loc: Literal["on data", "right"] = "on data",
):
    """Create a grid of UMAP plots across one or more datasets and features.

    Parameters
    ----------
    adata
        A single AnnData object or a list of AnnData objects to plot.
    features
        Features to visualize for each dataset.
    cmap
        Colormap or categorical color mapping passed through to :func:`umap`.
    umap_obsm
        Key in ``adata.obsm`` containing the embedding coordinates.
    legend_loc
        Legend placement strategy forwarded to :func:`umap`.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing all generated panels.
    """
    if isinstance(adata, AnnData):
        adata = [adata]
    
    ngraphs = len(adata) * len(features)
    w = 2 if ngraphs > 1 else 1
    h = int(np.ceil(ngraphs / w))
    fig, axes = plt.subplots(
        h, 2 if w == 1 else 5, figsize=(11 if w == 1 else 25, 10 * h), 
        gridspec_kw = {
            'width_ratios': [20, 1] if w == 1 else [20, 1, 5, 20, 1],
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
            ax = axes[y, x*3]
            side_ax = axes[y, x*3 + 1]
            if index < ngraphs:
                added_text.append(umap(
                    adata = adata[index // len(features)],
                    feature = features[index % len(features)], 
                    cmap = cmap,
                    umap_obsm_key = umap_obsm,
                    legend_loc = legend_loc, 
                    ax = ax, 
                    side_ax = side_ax,
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
    legend_kws: dict[str, Any] | None = None,
    legend_loc: Literal["horizontal", "vertical"] = "horizontal",
    legend_order: list | np.ndarray | None = None,
    figsize: tuple[int, int] | None = None,
    cmap: mplc.Colormap = DEFAULT_CMAP,
    s: float | None = None,
    a: float | None = None,
    ncol: int = 2,
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
    legend_kws
        Additional keyword arguments applied to the shared categorical legend.
    legend_loc
        Place the shared legend or colorbar below the plots or to their right.
    legend_order
        Optional explicit category ordering for categorical legends.
    figsize
        Figure size for the multi-panel layout.
    cmap
        Colormap used for continuous data.
    s
        Scatter marker size. Defaults to a size derived from the embedding extent.
    ncol
        Maximum number of subplot columns.
    vcenter
        Optional center value for diverging continuous color scales.
    **kwargs
        Additional keyword arguments forwarded to :func:`umap` for each panel.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the split UMAP panels and shared legend or colorbar.
    """
    fig = plt.figure(figsize=figsize or (10, 8), dpi=DPI)  # wider to fit legend or colorbar
    
    groups = adata.obs[group_key].unique().tolist()
    # adatas = [adata[adata.obs[group_key] == t] for t in groups]
    # if isinstance(features, str):
    #     features = [features]
    # return multiple_umap(adatas, features, **kwargs)
    
    w = min(ncol, len(groups))
    h = max(1, int(np.ceil(len(groups) / w)))

    axes, side_ax = subplots_with_side_axis(fig, h, w, legend_loc, legend_portion)
    side_ax.axis("off")
    
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
        
        kwargs = {"vmin": vmin, "vmax": vmax, "cmap": cmap} | kwargs
        if vcenter is not None:
            kwargs = {"vcenter": vcenter} | kwargs

    # Create each subplot
    for i, t in enumerate(groups):
        subgroup_mask = (adata.obs[group_key] == t).to_numpy()
        adata_sub = adata[subgroup_mask]
        adata_sub.uns = adata.uns
        
        kwargs_subplot = kwargs.copy()
        sub_bottom_points = bottom_mask[subgroup_mask] if bottom_mask is not None else None
        
        umap(
            adata=adata_sub,
            feature=feature,
            ax=axes[i],
            side_ax=None,
            umap_obsm_key=umap_obsm_key,
            legend_loc="right",
            s=s,
            a=a,
            xlim=xlim,
            ylim=ylim,
            bottom_points=sub_bottom_points,
            **kwargs_subplot,
        )
        axes[i].set_title(clean_title(rename(adata, str(t))))
        # axes[i].set_xlabel(None)
        # axes[i].set_ylabel(None)

    # Add either colorbar or legend
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
            ax=side_ax, 
            adata=adata, 
            feature=feature, 
            color_dict=legend_color_dict,
            legend_order=legend_order,
            loc=("upper center" if legend_loc == "horizontal" else "center left"),
            **(legend_kws or {}),
        )
    else:
        colorbar_title = clean_title(rename(adata, feature))
        if feature in adata.var_names:
            colorbar_title += " Expression"
        side_ax.axis("on")
        make_colorbar(
            sm=sm,
            cax=side_ax,
            label=colorbar_title,
            vmin=vmin,
            vmax=vmax,
            vcenter=vcenter,
            **(legend_kws or {}),
        )
        
    plt.tight_layout()
    return fig
