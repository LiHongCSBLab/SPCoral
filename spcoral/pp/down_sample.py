import numpy as np
import pandas as pd
import scanpy as sc
import anndata
from typing import Optional

__all__ = [
    'downsampling'
]

def downsampling(
        adata: anndata.AnnData,
        resolution: float,
        method: Optional[str] = 'sum',
        use_obsm: str = 'spatial',
        celltype_label: Optional[str] = None,
        drop_min: int = 0,
) -> anndata.AnnData:
    """
    Downsample a spatial omics dataset by binning spots/cells into a regular grid.

    This function aggregates expression data from individual spots or cells into rectangular
    bins defined by a fixed physical `resolution`. Cells are assigned to bins based on their spatial coordinates.
    Aggregation can be performed by summing counts (default) or averaging (`method='mean'`). Optionally,
    cell-type composition per bin can be recorded if a cell-type annotation column is provided.

    Parameters
    ----------
    adata : anndata.AnnData
        Input AnnData object containing spatial transcriptomics data. Spatial coordinates must be stored
        in ``obsm[use_obsm]`` (typically ``obsm['spatial']``).
    resolution : float
        Physical size of each bin (same units as spatial coordinates). The grid is aligned to the minimum
        coordinates and covers the full extent of the tissue.
    method : {'sum', 'mean'}, optional (default: 'sum')
        Aggregation method:
        - 'sum': total counts per gene in each bin.
        - 'mean': average expression per gene (total counts divided by number of cells/spots in the bin).
    use_obsm : str, optional (default: 'spatial')
        Key in ``.obsm`` where spatial coordinates (n_cells × 2) are stored.
    celltype_label : str, optional
        Column name in ``adata.obs`` containing cell-type annotations. If provided, a matrix of cell-type
        counts per bin will be stored in ``adata_bin.uns['cell_type']``.
    drop_min : int, optional (default: 0)
        Minimum number of cells/spots required in a bin for it to be retained. Bins with fewer cells are discarded.

    Returns
    -------
    anndata.AnnData
        Downsampled AnnData object where each observation corresponds to a spatial bin:
        - ``.X``: aggregated gene expression matrix (sum or mean).
        - ``.obsm['spatial']``: integer bin coordinates (grid indices).
        - ``.obsm['spatial_raw']``: physical center coordinates of each bin.
        - ``.obs['cell_counts']``: number of original cells/spots in each bin.
        - ``.uns['cell_type']`` (if ``celltype_label`` provided): DataFrame of cell-type counts per bin.

    """
    min_x = np.min(adata.obsm[use_obsm].T[0])
    min_y = np.min(adata.obsm[use_obsm].T[1])

    width = (np.max(adata.obsm[use_obsm].T[0]) - np.min(adata.obsm[use_obsm].T[0])) // resolution
    height = (np.max(adata.obsm[use_obsm].T[1]) - np.min(adata.obsm[use_obsm].T[1])) // resolution

    if celltype_label is not None:
        dict_result = {item: index for index, item in enumerate(list(adata.obs[celltype_label].unique()))}
        ct_data = np.zeros(((int(width) + 1) * (int(height) + 1), len(dict_result)))

    exp_data = np.zeros(((int(width) + 1) * (int(height) + 1), len(adata.var_names)))
    adata_spa = np.zeros(((int(width) + 1) * (int(height) + 1), 2))
    adata_spa_raw = np.zeros(((int(width) + 1) * (int(height) + 1), 2))
    count_data = np.zeros(((int(width) + 1) * (int(height) + 1), 1))

    for i in range(adata_spa.shape[0]):
        x = i - (i // (int(width) + 1)) * (int(width) + 1)
        y = i // (int(width) + 1)

        adata_spa[i, 0] = x
        adata_spa[i, 1] = y

        adata_spa_raw[i, 0] = x * resolution + min_x + resolution*0.5
        adata_spa_raw[i, 1] = y * resolution + min_y + resolution*0.5

    crds = adata.obsm[use_obsm]
    exp = adata.X

    if celltype_label is not None:
        cts = adata.obs[celltype_label].tolist()
    a = 0
    for x, y in crds:
        an_x = int((x - min_x) // resolution)
        an_y = int((y - min_y) // resolution)
        if celltype_label is not None:
            ct_data[an_y * (int(width) + 1) + an_x, dict_result[cts[a]]] += 1

        exp_data[an_y * (int(width) + 1) + an_x, :] += exp[a]
        count_data[an_y * (int(width) + 1) + an_x, 0] += 1
        a += 1

    adata_bin = sc.AnnData(exp_data, var=adata.var, obsm={'spatial': adata_spa, 'spatial_raw': adata_spa_raw})
    adata_bin = adata_bin[count_data.T[0] > drop_min, :].copy()

    if method == 'mean':
        adata_bin.X = adata_bin.X / count_data[count_data.T[0] > drop_min, :]

    adata_bin.obs['cell_counts'] = count_data.T[0][count_data.T[0] > drop_min]
    if celltype_label is not None:
        ct_data = pd.DataFrame(ct_data[count_data.T[0] > drop_min, :].copy(), index=adata_bin.obs_names,
                               columns=list(adata.obs[celltype_label].unique()))
        adata_bin.uns['cell_type'] = ct_data

    return adata_bin

if __name__ == '__main__':
    pass