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

For a fuller worked example, see the notebook in `example/scuva_test.ipynb`.

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

### Composition plots

`graph_counts` plots raw counts and `graph_proportions` plots percentages.

```python
scv.graph_counts(adata, hue="sample", x="leiden", stack=True)
scv.graph_proportions(adata, x="sample", y="leiden", figsize=(3, 6))
```

These functions return the summary table together with the figure and axes, which makes them easy to reuse in reports or downstream scripts.

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

- `umap(adata, feature, ...)`
- `multiple_umap(adata, features, ...)`
- `umap_split(adata, feature, group_key, ...)`
- `graph_counts(adata, hue, x, ...)`
- `graph_proportions(adata, x, y, ...)`

### Legends and color helpers

- `set_categorical_colors(adata, feature, color_mapping)`
- `get_categorical_colormap(adata, feature)`
- `make_legend(ax, title, label_color_dict, ...)`
- `make_colorbar(sm, cax, label, ...)`
- `subplots_with_side_axis(fig, nrows, ncols, side_ax_direction, side_ax_proportion)`

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
