import numpy as np
import pandas as pd
import scanpy as sc
import anndata

__all__=[
    'mapping_slides',
]

def find_nearest_points(target_point, points, k=5):

    distances = np.linalg.norm(points - target_point, axis=1)

    if np.min(distances) > 50:
        nearest_points_indices = None
    else:
        nearest_points_indices = np.argsort(distances)[:k]

    return nearest_points_indices

def mapping_slides(
    adata_slide1: anndata.AnnData,
    adata_slide2: anndata.AnnData,
    n_merge: int = 5,
    use_obsm: str = 'spatial',
) -> anndata.AnnData:
    """
    Map expression profiles from one spatial slide onto the geometry of another slide via nearest-neighbor averaging.

    Parameters
    ----------
    adata_slide1 : anndata.AnnData
        Target slide whose spatial coordinates will be used for the output.
        Spots in this object define the new grid positions.
    adata_slide2 : anndata.AnnData
        Source slide containing the expression/features to be transferred.
        Must have compatible spatial coordinates in ``obsm[use_obsm]``.
    n_merge : int, optional (default: 5)
        Number of nearest neighbors in ``adata_slide2`` to average for each spot in ``adata_slide1``.
    use_obsm : str, optional (default: 'spatial')
        Key in ``.obsm`` containing spatial coordinates (n_obs × 2 array).

    Returns
    -------
    anndata.AnnData
        New AnnData object with:
        - ``.X``: averaged expression matrix of shape (n_spots_slide1 × n_genes_slide2)
        - ``.obs``: observation names from ``adata_slide1``
        - ``.var``: variable (gene/feature) names from ``adata_slide2``
        - ``.obsm[use_obsm]``: spatial coordinates copied from ``adata_slide1``
        Spots with no valid neighbors (extremely rare) are filtered out.
    """
    meta = adata_slide2.X.T

    marker = np.zeros((adata_slide2.shape[1], adata_slide1.shape[0])).astype(np.float32)
    neigh_spots = adata_slide2.obsm[use_obsm].copy()

    selected_index = []

    for index, spot in enumerate(adata_slide1.obsm[use_obsm]):
        nearest_points_indices = find_nearest_points(spot, neigh_spots, k=n_merge)

        if nearest_points_indices is None:
            continue
        else:
            selected_index.append(index)
            marker[:, index] = np.mean(meta[:, nearest_points_indices], axis=1)

    adata_slide2_merge = sc.AnnData(marker.T, obs=pd.DataFrame(index=adata_slide1.obs_names.tolist()),
                                 var=pd.DataFrame(index=adata_slide2.var_names.tolist()), obsm={})
    adata_slide2_merge.obsm[use_obsm] = adata_slide1.obsm[use_obsm]
    sc.pp.filter_cells(adata_slide2_merge, min_counts=1)

    return adata_slide2_merge

if __name__ == '__main__':
    pass