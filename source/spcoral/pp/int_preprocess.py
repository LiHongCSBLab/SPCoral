import numpy as np
import anndata as ad
from typing import Tuple

__all__ = [
    'preprogress_adata',
    'extract_spatial_region',
]

def create_snn_adjacency_matrix(coords1, coords2, k_neighbors):
    """
    Generate a Shared Nearest Neighbor (SNN) adjacency matrix where each row represents
    the SNN neighbors from coords2 for each point in coords1.

    Args:
        coords1: numpy array of shape (m, d), first set of coordinates
        coords2: numpy array of shape (n, d), second set of coordinates
        k_neighbors: int, number of nearest neighbors to consider for SNN

    Returns:
        adjacency_matrix: numpy array of shape (m, n), binary matrix where
                         1 indicates a shared nearest neighbor relationship
    """
    m, n = len(coords1), len(coords2)

    distances_1_to_2 = np.sqrt(((coords1[:, np.newaxis] - coords2) ** 2).sum(axis=2))
    distances_2_to_1 = distances_1_to_2.T

    knn_1_to_2 = np.argsort(distances_1_to_2, axis=1)[:, :k_neighbors]  # coords1 -> coords2
    knn_2_to_1 = np.argsort(distances_2_to_1, axis=1)[:, :k_neighbors]  # coords2 -> coords1

    adjacency_matrix = np.zeros((m, n), dtype=int)

    for i in range(m):
        for j in range(n):
            # Point i from coords1 and point j from coords2 are SNN if:
            # j is in i's k-nearest neighbors AND i is in j's k-nearest neighbors
            if j in knn_1_to_2[i] and i in knn_2_to_1[j]:
                adjacency_matrix[i, j] = 1

    return adjacency_matrix


def row_normalize(A):
    row_sums = A.sum(axis=1)
    normalized_A = np.zeros_like(A, dtype=float)
    for i in range(A.shape[0]):
        if row_sums[i] != 0:
            normalized_A[i, :] = A[i, :] / row_sums[i]
    return normalized_A


def preprogress_adata(
    adata_omics1: ad.AnnData,
    adata_omics2: ad.AnnData,
    method: str = 'knn',
    k: int = 10,
    use_obsm: str = 'spatial',
) -> Tuple[ad.AnnData, ad.AnnData]:
    """
    Preprocess two spatial omics AnnData objects by removing spots/cells with no cross-omics spatial neighbors.

    Parameters
    ----------
    adata_omics1 : anndata.AnnData
        First spatial omics AnnData object. Must contain spatial coordinates in ``obsm[use_obsm]``.
    adata_omics2 : anndata.AnnData
        Second spatial omics AnnData object with the same spatial coordinate requirement.
    method : {'knn'}, optional (default: 'knn')
        Method used to define spatial neighbors across datasets. Currently only 'knn' is supported.
    k : int, optional (default: 10)
        Number of nearest neighbors in the other omics to consider when building the cross-omics adjacency.
        Each spot in omics1 finds its `k` nearest spots in omics2, and vice versa (resulting in a bipartite graph).
    use_obsm : str, optional (default: 'spatial')
        Key in ``.obsm`` where spatial coordinates (n_obs × 2) are stored.

    Returns
    -------
    tuple of (anndata.AnnData, anndata.AnnData)
        - Filtered copy of ``adata_omics1`` containing only spots with at least one neighbor in omics2.
        - Filtered copy of ``adata_omics2`` containing only spots with at least one neighbor in omics1.

    """

    crd_omics1 = adata_omics1.obsm[use_obsm]
    crd_omics2 = adata_omics2.obsm[use_obsm]

    if method == 'knn':
        g_crd_cross = create_snn_adjacency_matrix(
            crd_omics1, crd_omics2, k
        ).astype(np.float32)

    adata_omics1_sub = adata_omics1[np.sum(g_crd_cross, axis=1) > 0, :].copy()
    adata_omics2_sub = adata_omics2[np.sum(g_crd_cross, axis=0) > 0, :].copy()

    return adata_omics1_sub, adata_omics2_sub


def extract_spatial_region(
        adata: ad.AnnData,
        minx: float,
        miny: float,
        maxx: float,
        maxy: float,
        retainx = 0, retainy = 0,
        used_obsm: str = 'spatial'
) -> ad.AnnData:
    """
    Extract a spatial region from an AnnData object based on spatial coordinates.

    Parameters:
    -----------
    adata : anndata.AnnData
        The input AnnData object containing spatial transcriptomics data.
    minx : float
        Minimum x-coordinate of the region.
    miny : float
        Minimum y-coordinate of the region.
    maxx : float
        Maximum x-coordinate of the region.
    maxy : float
        Maximum y-coordinate of the region.
    used_obsm : str
        The key in adata.obsm where spatial coordinates are stored (e.g., 'spatial').

    Returns:
    --------
    anndata.AnnData
        A new AnnData object containing only the data from the specified spatial region.
    """
    # Extract spatial coordinates from obsm
    coords = adata.obsm[used_obsm]

    # Ensure coordinates are 2D (x, y)
    if coords.shape[1] < 2:
        raise ValueError("Spatial coordinates in obsm must have at least 2 dimensions (x, y).")

    # Create boolean mask for the region
    mask = (coords[:, 0] >= minx - retainx) & (coords[:, 0] < maxx + retainx) & \
           (coords[:, 1] >= miny - retainy) & (coords[:, 1] < maxy + retainy)


    # Subset the AnnData object using the mask
    adata_subset = adata[mask].copy()
    if retainx:
        coords = adata_subset.obsm[used_obsm]

        mask_t = (coords[:, 0] >= minx) & (coords[:, 0] < maxx) & \
                 (coords[:, 1] >= miny) & (coords[:, 1] < maxy)
        adata_subset.obs['retain'] = False
        adata_subset.obs.loc[mask_t, 'retain'] = True

        adata_subset.obs['mask_up'] = False
        adata_subset.obs['mask_down'] = False
        adata_subset.obs['mask_right'] = False
        adata_subset.obs['mask_left'] = False

        mask_up = (coords[:, 1] >= maxy - retainy) & (coords[:, 1] < maxy + retainy) & (coords[:, 0] >= minx - retainx) & (coords[:, 0] < maxx + retainx)
        mask_down = (coords[:, 1] >= miny - retainy) & (coords[:, 1] < miny + retainy) & (coords[:, 0] >= minx - retainx) & (coords[:, 0] < maxx + retainx)
        mask_right = (coords[:, 1] >= miny - retainy) & (coords[:, 1] < maxy + retainy) &  (coords[:, 0] >= maxx - retainx) & (coords[:, 0] < maxx + retainx)
        mask_left = (coords[:, 1] >= miny - retainy) & (coords[:, 1] < maxy + retainy) & (coords[:, 0] >= minx - retainx) & (coords[:, 0] < minx + retainx)


        adata_subset.obs.loc[mask_up, 'mask_up'] = True
        adata_subset.obs.loc[mask_down, 'mask_down'] = True
        adata_subset.obs.loc[mask_right, 'mask_right'] = True
        adata_subset.obs.loc[mask_left, 'mask_left'] = True

    else:
        adata_subset.obs['retain'] = True
        adata_subset.obs['mask_up'] = False
        adata_subset.obs['mask_down'] = False
        adata_subset.obs['mask_right'] = False
        adata_subset.obs['mask_left'] = False

    return adata_subset