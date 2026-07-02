import numpy as np
from scipy.stats import median_abs_deviation
from anndata import AnnData


"""
This code is adapted from the single-cell best practices online guide https://www.sc-best-practices.org/preprocessing_visualization/quality_control.html.
Using MADs for outlier detection in single-cell is from Germain et al. https://doi.org/10.1186/s13059-020-02136-7.
"""
def is_outlier(adata: AnnData, metric: str, n_mads: int) -> np.ndarray:
    if "outlier_mad_thresholds" not in adata.uns:
        adata.uns["outlier_mad_thresholds"] = dict()
        
    M = adata.obs[metric]
    median = np.median(M)
    MADs = n_mads * median_abs_deviation(M)
    if isinstance(adata.uns["outlier_mad_thresholds"], dict):
        adata.uns["outlier_mad_thresholds"][metric] = [median - MADs, median + MADs]
    outlier = (M < median - MADs) | (median + MADs < M)
    return outlier
