# scuva

**S**ingle **C**ell **U**tility for **V**isualization and **A**nalysis

`scuva` is a plotting helper library for `scanpy` and `anndata` workflows. It focuses on the repetitive parts of single-cell figure-making: consistent UMAP styling, categorical color management, shared legends/colorbars, and compact composition plots.

## What it does

`scuva` currently provides three main groups of helpers:

| Area | Functions | Purpose |
| --- | --- | --- |
| UMAP plotting | `umap`, `multiple_umap`, `umap_split` | Plot categorical or continuous features from an `AnnData` object with consistent legends and colorbars. |
| Composition plotting | `graph_counts`, `graph_proportions` | Summarize cell counts or percentages across observation columns. |
| Text and color utilities | `set_categorical_colors`, `get_categorical_colormap`, `make_legend`, `make_colorbar`, `rename`, `clean_title`, `wrap_join` | Keep labels, legends, and category colors readable and consistent. |

## Installation

```bash
pip install scuva
```

The package metadata currently lists these runtime dependencies:

- `anndata`
- `matplotlib`
- `numpy`
- `pandas`
- `scanpy`

## Expected AnnData conventions

`scuva` relies on the `AnnData` object to follow `scanpy` conventions:

- UMAP coordinates live in `adata.obsm["X_umap"]` unless you pass a different `umap_obsm_key`.
- Categorical plotting functions expect the relevant `adata.obs` column to use a pandas categorical dtype.
- Category colors are stored in the `adata.uns[f"{feature}_colors"]` entry.
- Optional display renaming can be stored in `adata.uns["rename_dict"]`.
- `graph_proportions`, `set_categorical_colors`, and most categorical legend helpers expect the relevant observation columns to be pandas categoricals.
- `graph_counts` also expects `hue` to be categorical because colors are read from `adata.obs[hue].cat.categories`.
- `get_expression_by_obs` forwards its `layer` argument to `scanpy.tl.rank_genes_groups` and uses the same matrix again when attaching summary columns.

If a helper depends on one of these conventions, the function usually raises a `ValueError` with a direct explanation when the input does not match.

## Quick start

```python
import scanpy as sc
import scuva as scv

adata = sc.read_10x_mtx("test/data/pbmc3k/hg19")

adata.layers["counts"] = adata.X.copy()
sc.pp.normalize_total(adata, target_sum=10_000)
sc.pp.log1p(adata)

sc.pp.highly_variable_genes(
	adata,
	flavor="seurat_v3",
	n_top_genes=2000,
	layer="counts",
)
sc.tl.pca(adata, svd_solver="arpack", use_highly_variable=True)
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=8, use_rep="X_pca")
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.5)

scv.umap(adata, "leiden", legend_loc="right")
```

For a fuller worked example, see the notebook in `test/scuva_test.ipynb`.

## Core workflows

### UMAP plots

Use `umap` for a single panel.

```python
scv.umap(adata, "leiden", legend_loc="right")
scv.umap(adata, "MS4A1", vcenter=0, legend_loc="right")
```

Behavior to be aware of:

- If `feature` is categorical, `scuva` uses categorical colors and draws either a side legend or labels directly on the embedding.
- If `feature` is continuous, `scuva` builds a colorbar from the value range and treats zeros as background when plotting the colored layer.
- `bottom_points` can be a boolean mask or an array of indices and is useful for drawing selected cells underneath the rest.
- `layer` and `use_raw=True` are mutually exclusive.

Use `multiple_umap` to compare several features or several `AnnData` objects in one figure.

```python
scv.multiple_umap(adata, ["leiden", "MS4A1"], legend_loc="right")
```

Current behavior to keep in mind:

- `legend_loc` may be a single value or a list that is cycled across panels.
- `multiple_umap` currently lays out at most two plotted panels per row reliably, so `ncols` should stay at `1` or `2`.
- `bottom_points` is available on `umap`, but `multiple_umap` does not expose that control to individual panels.

Use `umap_split` to create one panel per group.

```python
scv.umap_split(
	adata,
	feature="leiden",
	group_key="sample",
	legend_loc="vertical",
	figsize=(10, 4),
)
```

`umap_split` uses the order returned by `adata.obs[group_key].unique()` when building panels. For continuous features it applies one shared color scale across all panels. For categorical features, the subplot colors come from the usual AnnData-stored or default category colors.

### Composition plots

`graph_counts` plots raw counts and `graph_proportions` plots percentages.

```python
scv.graph_counts(adata, hue="sample", x="leiden", stack=True)
scv.graph_proportions(adata, x="sample", y="leiden", figsize=(3, 6))
```

These functions return the summary table together with the figure and axes, which makes them easy to reuse in reports or downstream scripts.

### Marker inspection

Use `get_expression_by_obs` to run `scanpy.tl.rank_genes_groups` and attach two summary columns to the ranked-gene output:

- `percent_expressing`: percentage of cells with nonzero raw counts.
- `average_expression`: `log1p(mean(expression))` for the requested expression source.

If `layer=None`, the function assumes `adata.X` contains integer count data and does not reverse any prior `log1p` transform.
For a typical Scanpy workflow where `adata.X` contains normalized values, pass the layer name that contains raw integer count data.

```python
markers = scv.get_expression_by_obs(adata, "leiden", layer="counts")

scv.check_expression(
	adata,
	markers,
	features=["MS4A1", "CD79A"],
	cluster_column="leiden",
	score_threshold=2,
)
```

`check_expression` prints the selected marker rows and then draws a small UMAP panel set containing the grouping column plus each requested gene.

### Colors and labels

You can override category colors explicitly:

```python
scv.set_categorical_colors(
	adata,
	"leiden",
	{
		"0": "red",
		"1": "orange",
		"2": "blue",
	},
)
```

You can also provide a shared renaming dictionary for display text:

```python
adata.uns["rename_dict"] = {
	"leiden": "cluster",
	"MS4A1": "CD20",
	"sample_0": "control",
}
```

That renaming is applied by helpers such as `rename`, `umap`, `graph_counts`, and `graph_proportions` when generating titles and labels.

## Public API

### Plotting

- `umap(adata, feature, use_raw=False, layer=None, legend_loc="on data", vmin=None, vmax=None, vcenter=None, bottom_points=None, ...)`
- `multiple_umap(adata, features, layer=None, ncols=2, legend_loc="on data", ...)`
- `umap_split(adata, feature, group_key, legend_orientation="horizontal", ncols=2, ...)`
- `graph_counts(adata, hue, x, stack=False, sort_by_size=True, ...)`
- `graph_proportions(adata, x, y, x_order="sort", combine_small_percentages_other=False, ...)`

### Analysis

- `get_expression_by_obs(adata, column, layer="counts")`
- `check_expression(adata, expression_by_obs_df, features, cluster_column="cluster", cluster_subset=None, score_threshold=None)`

### Legends and color helpers

- `set_categorical_colors(adata, feature, color_mapping)`
- `get_categorical_colormap(adata, feature)`
- `make_legend(ax, title, label_color_dict, sort_ints="forward", ...)`
- `make_colorbar(sm, cax, label, orientation="vertical", ...)`
- `subplots_with_legend_axis(fig, total_subplots, nrows, ncols, side_ax_orientation, side_ax_proportion, use_extra_subplot_axis=True)`

### Text helpers

- `wrap_join(items, sep=" ", width=30)`
- `rename(adata, t, additional_renaming=None)`
- `clean_title(s)`

## Scope and limitations

This project is still small and evolving. A few details are worth calling out explicitly:

- The API is oriented around exploratory plotting rather than a full declarative plotting system.
- The plotting helpers assume a 2D UMAP-like embedding and do not try to generalize to arbitrary coordinate systems.
- The package leans on Scanpy conventions instead of re-implementing its own metadata model.

## Contributing

Issues and pull requests are welcome. The project is not yet stable. Parts of the API may still change.

## License

This code is licensed under GPL-3.0-or-later. See `LICENSE`.
