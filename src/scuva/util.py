"""Internal validation and sorting helpers used across the package."""
from __future__ import annotations


from pandas import Index, CategoricalDtype
from numpy import ndarray, issubdtype, number
from numbers import Number


def _require_categorical(adata, obs_column):
    """Require that ``adata.obs[obs_column]`` exists and is categorical.

    Parameters
    ----------
    adata
        Object exposing an ``obs`` DataFrame-like attribute.
    obs_column
        Observation column that must be present and use a pandas categorical dtype.

    Raises
    ------
    ValueError
        If the column is missing or is not categorical.
    """
    if obs_column not in adata.obs.columns:
        raise ValueError(f"'{obs_column}' not found in adata.obs")
    if not isinstance(adata.obs[obs_column].dtype, CategoricalDtype):
        raise ValueError(f"Column '{obs_column}' is not categorical. Check that the column data is correct, then cast the column to categorical:\n"
                         + f"adata.obs[{obs_column}] = adata.obs[{obs_column}].astype('category')")


def is_numeric(arr: ndarray):
    """Return ``True`` when an array stores only numeric values.

    Native numeric dtypes are accepted immediately. Object arrays are accepted only
    when every element is an instance of :class:`numbers.Number`.
    """
    # Fast path for native numeric dtypes
    if issubdtype(arr.dtype, number):
        return True
    # If array has object dtype, check element-wise for Python numeric types
    if arr.dtype == object:
        return all(isinstance(x, Number) for x in arr.flat)
    return False


def sort_categories_handle_ints(
    categories: list[str] | Index
) -> list[str]:
    """Sort category labels numerically when possible, otherwise lexicographically.

    Pandas ``Index`` inputs are first converted to strings so integer-like category
    labels such as ``"10"`` and ``"2"`` can be sorted numerically.
    """
    if isinstance(categories, Index):
        categories = categories.astype(str).to_list()
    try:
        return sorted(categories, key=int)
    except (TypeError, ValueError):
        return sorted(categories) 
