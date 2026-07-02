import scanpy as sc
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from ..text import clean_placeholder_string
from textwrap import wrap

# TODO: generalize this code better
# TODO: combine the clean placeholder string with the existing renaming system

clean_metric_names = {
    "pct_counts_mt": "% Reads in Mitochondrial Genes",
    "pct_counts_in_top_$X_genes": "% Reads in $X Most Expressed Genes",
    "total_counts": "Total Reads",
    "n_genes_by_counts": "Unique Genes Detected",
}

def qc_metrics_violinplot(adata, metrics: list[str]) -> Axes:
    ax = sc.pl.violin(adata, metrics, show=False)
    assert isinstance(ax, Axes)
    
    metric_names = ["\n".join(wrap(clean_placeholder_string(m, clean_metric_names), 25)) for m in metrics]
    t = adata.uns["outlier_mad_thresholds"]
    
    ax.set_xticklabels(metric_names, fontsize=14)
    if all(m.startswith("pct_counts_") for m in metrics):
        ax.set_ylabel("Percent of Reads", fontsize=14)
    else:
        ax.set_ylabel("Value", fontsize=14)
    ax.set_title("QC Metrics per Cell", fontsize=14)
    
    for i, m in enumerate(metrics):
        ax.plot((i - 0.4, i + 0.4), [t[m][0]]*2, color="orange", ls="--")
        ax.plot((i - 0.4, i + 0.4), [t[m][1]]*2, color="orange", ls="--")
    
    return ax


# TODO: generalize this code better. Currently only supports total_counts for x, n_genes_by_counts for y, and expects log1p-based threshold values

def qc_metrics_scatterplot(adata, color: str, color_limit: float | None = None, log: bool = True) -> Axes:
    ax = sc.pl.scatter(adata, "total_counts", "n_genes_by_counts", color=color, show=False)
    assert isinstance(ax, Axes)
    
    if color_limit is not None:
        ax.collections[0].set_clim(vmax=color_limit)
        cbar = ax.collections[0].colorbar
        assert isinstance(cbar, Colorbar)
        ticks = list(cbar.get_ticks())
        labels = [str(int(t)) for t in ticks[:-1]] + [f">{int(ticks[-1])}"]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(labels)
    
    metric_names = [clean_placeholder_string(m, clean_metric_names) for m in ["total_counts", "n_genes_by_counts", color]]
    ax.set_title(metric_names[2] + " per cell", fontsize=14)
    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    if "outlier_mad_thresholds" in adata.uns:
        t = adata.uns["outlier_mad_thresholds"]
        if "log1p_n_genes_by_counts" in t:
            # axes can be displayed on a log scale but they always use normal units
            ax.axhline(np.exp(t["log1p_n_genes_by_counts"][0])-1, color="orange", ls="--")
            ax.axhline(np.exp(t["log1p_n_genes_by_counts"][1])-1, color="orange", ls="--")
        if "log1p_total_counts" in t:
            # axes can be displayed on a log scale but they always use normal units
            ax.axvline(np.exp(t["log1p_total_counts"][0])-1, color="orange", ls="--")
            ax.axvline(np.exp(t["log1p_total_counts"][1])-1, color="orange", ls="--")
    ax.set_xlabel(metric_names[0], fontsize=14)
    ax.set_ylabel(metric_names[1], fontsize=14)
    
    return ax
