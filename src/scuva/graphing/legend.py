from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.colorbar import Colorbar
from matplotlib.legend import Legend
import numpy as np
import numpy.typing as npt
import textwrap

def make_colorbar(
    sm: ScalarMappable, 
    cax: Axes, 
    label: str,
    ticks: npt.ArrayLike | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    vcenter: float | None = None,
    label_fontsize: float = 14,
    tick_fontsize: float = 10,
    **kwargs
) -> Colorbar:
    """Draw a colorbar with consistent labeling and endpoint tick handling.

    Parameters
    ----------
    sm
        Scalar mappable used to generate the color scale.
    cax
        Axes that should contain the colorbar.
    label
        Text label shown alongside the colorbar.
    ticks
        Optional explicit tick locations. If omitted, ticks are inferred from the
        colorbar and adjusted so the upper endpoint is labeled cleanly.
    vmin, vmax, vcenter
        Data range used by the colorbar. ``vcenter`` is accepted for API
        compatibility with callers, although the norm should already be encoded on
        ``sm``.
    label_fontsize
        Font size for the colorbar label.
    tick_fontsize
        Font size for tick labels.
    **kwargs
        Additional keyword arguments forwarded to ``matplotlib.pyplot.colorbar``.
    """
    cbar = plt.colorbar(sm, cax=cax, **kwargs)
    cbar.set_label(label, fontsize=label_fontsize)
    cbar.ax.tick_params(labelsize=tick_fontsize)
    if ticks is not None:
        cbar.set_ticks(np.asarray(ticks, dtype=float).tolist())
    elif vmin is not None and vmax is not None:
        inferred_ticks = list(cbar.get_ticks())
        if len(inferred_ticks) >= 3:
            # Remove the last tick if it is too close to the maximum value to avoid
            # overlapping endpoint labels.
            if vmax - inferred_ticks[-2] < (inferred_ticks[-2] - inferred_ticks[-3]) / 4:
                inferred_ticks = inferred_ticks[:-2] + [vmax]
            else:
                inferred_ticks = inferred_ticks[:-1] + [vmax]
            cbar.set_ticks(inferred_ticks, labels=[f"{tick:.2f}" for tick in inferred_ticks])
    if vmin is not None and vmax is not None:
        cbar.ax.set_ylim(vmin, vmax)
    return cbar


def make_legend(
    ax: Axes,
    title: str,
    label_color_dict: dict,
    sort_ints: bool = True,
    label_rename_dict: dict | None = None,
    loc: str = "center left",
    markersize: float = 6,
    fontsize: int = 12,
    title_fontsize: int = 14,
    text_wrap_chars: int | None = 15,
) -> Legend:
    """Draw a simple square-marker legend on a dedicated axis.

    Parameters
    ----------
    ax
        Axis that should contain only the legend.
    title
        Legend title.
    label_color_dict
        Mapping from category label to marker color.
    sort_ints
        Sort labels numerically when they look like integers.
    label_rename_dict
        Optional display-name overrides for legend labels.
    loc
        Matplotlib legend location string.
    markersize
        Size of the square legend markers.
    fontsize
        Font size for legend labels.
    title_fontsize
        Font size for the legend title.
    text_wrap_chars
        Optional wrap width applied to the title.

    Returns
    -------
    matplotlib.legend.Legend
        The created legend object.
    """
    label_order = list(label_color_dict.keys())
    if sort_ints:
        try:
            label_order = sorted(label_order, key=int)
        except (TypeError, ValueError):
            pass
    
    if label_rename_dict is None:
        label_rename_dict = {k: k for k in label_color_dict.keys()}
    else:
        label_rename_dict = {k: k for k in label_color_dict.keys()} | label_rename_dict
    handles = [
        Line2D([], [], marker='s', linestyle='None', color=label_color_dict[label],
                label=label_rename_dict[label], markersize=markersize)
        for label in label_order
    ]
    
    if text_wrap_chars is not None:
        title = textwrap.fill(title, text_wrap_chars)
    
    legend = ax.legend(
        handles=handles,
        title = title,
        loc = loc,
        fontsize = fontsize,
        title_fontsize = title_fontsize
    )
    legend.get_title().set_multialignment('center')
    ax.set_facecolor('none')  # No background color
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.patch.set_alpha(0)
    return legend
