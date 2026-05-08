"""Marker-ranking helpers that summarize expression by observation groups."""

from __future__ import annotations

import scanpy as sc
from anndata import AnnData
import pandas as pd
import numpy as np
from scipy import sparse

from .graphing.umap import multiple_umap


def _flatten_1d(values) -> np.ndarray:
    """Convert row or column vector outputs into a flat NumPy array."""
    return np.asarray(values).reshape(-1)

def get_expression_by_obs(adata: AnnData, column: str, layer: str | None = "counts"):
    """Run ``rank_genes_groups`` and join matrix-derived summary columns.

    Parameters
    ----------
    adata
        AnnData object containing grouped observations and expression values.
    column
        Observation column passed to :func:`scanpy.tl.rank_genes_groups`.
    layer
        Layer name forwarded to :func:`scanpy.tl.rank_genes_groups` and used to
        select the matrix for the appended summary columns. When ``None``, the
        function uses ``adata.X`` directly.

    Returns
    -------
    pandas.DataFrame
        The Scanpy differential-expression table augmented with
        ``percent_expressing`` and ``average_expression``.

    Notes
    -----
    ``percent_expressing`` and ``average_expression`` are both computed from the provided layer or anndata.X.
    ``average_expression`` is reported on a ``log1p`` scale as log1p(mean(expression))``.
    When ``layer=None``, adata.X is still assumed to contain raw integer counts. No inverse transform is applied even if ``adata.X`` already contains log-transformed values.
    """
    if layer is None:
        if adata.X is None:
            raise ValueError("No data is present in adata.X")
    else:
        if layer not in adata.layers:
            raise ValueError("Expected adata.layers['counts'] to be present")

    sc.tl.rank_genes_groups(adata, groupby=column, method="wilcoxon", layer=layer)
    df = sc.get.rank_genes_groups_df(adata, group=None)
    df.rename(columns={"group": column}, inplace=True)
    df_pct_expr = []
    for c in df[column].unique():
        subset = adata[adata.obs[column] == c]
        matrix = subset.layers[layer] if layer is not None else subset.X
        percent_expressing = ((matrix > 0).sum(axis=0) / len(subset)) * 100
        percent_expressing = _flatten_1d(percent_expressing)
        average_expression = np.log1p(matrix.mean(axis=0))
        average_expression = _flatten_1d(average_expression)
        
        df_pct_expr.append(pd.DataFrame({
            column: c,
            "names": subset.var_names, 
            "percent_expressing": percent_expressing,
            "average_expression": average_expression,
        }).set_index([column, "names"]))
        
    df = df.join(pd.concat(df_pct_expr), on=[column, "names"], how="left")
    if pd.to_numeric(df[column], errors="coerce").notna().all():
        df = df.astype({column: int})
    return df.sort_values([column, "scores"], ascending=[True, False])

#? TODO: allow features from obs?
# TODO: fix to allow arbitrary key for filtering (not just clusters)
def check_expression(
    adata: AnnData, 
    expression_by_obs_df: pd.DataFrame,
    features: list[str] | str, 
    cluster_column: str = "cluster",
    cluster_subset: list[str] | None = None,
    score_threshold: float | None = None, 
):
    """Print filtered marker rows and plot the requested features on UMAP.

    Parameters
    ----------
    adata
        AnnData object used for plotting and feature validation.
    expression_by_obs_df
        Output from :func:`get_expression_by_obs`.
    features
        One or more genes to display in the summary table and UMAP grid.
    cluster_column
        Observation column representing the group labels in both ``adata`` and the
        summary DataFrame.
    cluster_subset
        Optional subset of cluster labels to include.
    score_threshold
        Optional lower bound applied to the differential-expression score.

    Returns
    -------
    None
        The function prints a filtered summary table and calls
        :func:`scuva.graphing.umap.multiple_umap` for side-effect plotting.

    """
    
    all_genes = set(adata.var_names)
    e = expression_by_obs_df
    
    if isinstance(features, str):
        features = [features]
    
    for gene in features:
        if gene not in all_genes:
            print("Gene not present:", gene)
    
    if cluster_subset is not None:
        str_clusters = [str(c) for c in cluster_subset]
        e = e.loc[e[cluster_column].astype(str).isin(str_clusters)]
        subset = adata[adata.obs[cluster_column].astype(str).isin(str_clusters)]
    else:
        subset = adata
    
    if score_threshold is not None:
        e = e.loc[(e.scores >= score_threshold)]
    
    print(e.loc[e.names.isin(features)]
        .sort_values([cluster_column, "scores"], ascending=[True, False])
        .loc[:, [cluster_column, "names", "scores", "percent_expressing", "average_expression"]])
    multiple_umap(
        subset, 
        features=[cluster_column] + features,
        ncols=3,
        legend_loc="on data",
        vmin=0,
    )
