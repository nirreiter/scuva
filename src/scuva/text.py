from __future__ import annotations

from anndata import AnnData

def wrap_join(items, sep=" ", width=30) -> str:
    """Join strings while wrapping to a maximum line width.

    Parameters
    ----------
    items
        Ordered strings to join.
    sep
        Separator inserted between adjacent items.
    width
        Maximum target width for each generated line.

    Returns
    -------
    str
        A newline-delimited string whose lines are wrapped at item boundaries.
    """
    if len(items) == 0:
        return ""

    lines = []
    current = items[0]

    for item in items[1:]:
        if len(current) + len(sep) + len(item) > width:
            lines.append(current)
            current = item
        else:
            current += sep + item
    lines.append(current)
    
    return "\n".join(lines)

def clean_title(s: str) -> str:
    """Convert an identifier-like string into a simple display title.

    Underscores are replaced with spaces and each word is capitalized.
    """
    words = [word for word in s.replace("_", " ").split() if word]
    return " ".join(word[:1].upper() + word[1:] for word in words)


def rename(adata: AnnData, t: str, additional_renaming: dict[str, str] | None = None) -> str:
    """Resolve a display label using explicit and AnnData-stored renaming rules.

    Parameters
    ----------
    adata
        AnnData object that may contain a ``rename_dict`` entry in ``adata.uns``.
    t
        Original label to resolve.
    additional_renaming
        Optional mapping that takes precedence over any mapping stored on ``adata``.

    Returns
    -------
    str
        The renamed label if a mapping exists, otherwise the original value.
    """
    if additional_renaming is not None and t in additional_renaming:
        return additional_renaming[t]
    if adata is None or "rename_dict" not in adata.uns or t not in adata.uns["rename_dict"]:
        return t
    return adata.uns["rename_dict"][t]
