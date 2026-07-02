"""Marker-ranking helpers that summarize expression by observation groups."""

from __future__ import annotations

import scanpy as sc
from anndata import AnnData
import pandas as pd
import numpy as np
from scipy import sparse

from ..graphing.umap import multiple_umap


def _flatten_1d(values) -> np.ndarray:
    """Convert row or column vector outputs into a flat NumPy array."""
    return np.asarray(values).reshape(-1)

def get_expression_by_obs(
    adata: AnnData, 
    column: str, 
    integer_layer: str | None = "counts",
    normalized_layer: str | None = None,
    scoring_layer: str | None = None,
):
    """Run ``rank_genes_groups`` and join matrix-derived summary columns.

    Parameters
    ----------
    adata
        AnnData object containing grouped observations and expression values.
    column
        Observation column passed to :func:`scanpy.tl.rank_genes_groups`.
    integer_layer
        Layer used to compute ``percent_expressing``. This layer should contain
        unnormalized integer counts. When ``None``, the function uses
        ``adata.raw.X`` if available, otherwise ``adata.X``.
    normalized_layer
        Layer used to compute ``average_expression``. When ``None``, the function
        uses ``adata.raw.X`` if available, otherwise ``adata.X``.
    scoring_layer
        Layer name forwarded to :func:`scanpy.tl.rank_genes_groups`.

    Returns
    -------
    pandas.DataFrame
        The Scanpy differential-expression table augmented with
        ``percent_expressing`` and ``average_expression``.

    Notes
    -----
    When ``scoring_layer`` is ``None`` and ``adata.raw`` exists, gene scores are
    computed from ``adata.raw``
    ``percent_expressing`` and ``average_expression`` are computed separately from
    ``integer_layer`` and ``normalized_layer``.
    ``average_expression`` is calculated on a linear scale but reported on a ``log1p`` scale.
    When either summary-layer argument is ``None``, the fallback matrix is
    ``adata.raw.X`` if present, otherwise ``adata.X``.
    """
    if integer_layer is None:
        if adata.X is None:
            raise ValueError("No data is present in adata.X")
    else:
        if integer_layer not in adata.layers:
            raise ValueError(f"Expected adata.layers['{integer_layer}'] to be present")
    if normalized_layer is None:
        if adata.X is None:
            raise ValueError("No data is present in adata.X")
    else:
        if normalized_layer not in adata.layers:
            raise ValueError(f"Expected adata.layers['{normalized_layer}'] to be present")
    
    # TODO: Add a check that the provided layer / adata.X is integer data

    sc.tl.rank_genes_groups(adata, groupby=column, method="wilcoxon", 
                            scoring_layer=scoring_layer, use_raw=(scoring_layer is None and adata.raw is not None))
    df = sc.get.rank_genes_groups_df(adata, group=None)
    df.rename(columns={"group": column}, inplace=True)
    df_pct_expr = []
    for c in df[column].unique():
        subset = adata[adata.obs[column] == c]
        subset_main_matrix = subset.raw.X if subset.raw else subset.X
        integer_matrix = subset.layers[integer_layer] if integer_layer is not None else subset_main_matrix
        percent_expressing = ((integer_matrix > 0).sum(axis=0) / len(subset)) * 100
        percent_expressing = _flatten_1d(percent_expressing)
        normalized_matrix = subset.layers[normalized_layer] if normalized_layer is not None else subset_main_matrix
        average_expression = np.log1p(np.expm1(normalized_matrix).mean(axis=0))
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

#? TODO: allow features from_obs?
# TODO: fix to allow arbitrary key for filtering (not just clusters)
def check_expression(
    adata: AnnData, 
    expression_by_obs_df: pd.DataFrame,
    features: list[str] | str, 
    cluster_column: str = "cluster",
    cluster_subset: list[str] | None = None,
    score_threshold: float | None = None, 
):
    """Filter a marker table and plot the requested features on UMAP.

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
    tuple[Figure, pandas.DataFrame]
        The generated figure together with the filtered summary table.

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
    
    df = (
        e.loc[e.names.isin(features)]
        .sort_values([cluster_column, "scores"], ascending=[True, False])
        .loc[:, [cluster_column, "names", "scores", "percent_expressing", "average_expression"]]
    )
        
    fig = multiple_umap(
        subset, 
        features=[cluster_column] + features,
        ncols=2,
        legend_loc="on data",
        vmin=0,
    )
    
    return fig, df
