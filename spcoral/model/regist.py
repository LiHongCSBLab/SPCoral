# import
import anndata
from typing import Tuple, List, Any

from .utils import FGW_OT, find_anchor, apply_transform, compute_affine_transform

__all__ = [
    'registration',
    'registration_by_downsampling'
]

def registration(
    adata_omics1: anndata.AnnData,
    adata_omics2: anndata.AnnData,
    alpha: float = 1.0,
    beta: float = 0.8,
    n_iter: int = 10,
    method: str = 'affine',
    use_obsm: str = 'spatial',
    new_obsm: str = 'spatial_reg',
    random_state: int = 2030,
) -> Tuple[anndata.AnnData, List[Any]]:
    """
    Perform spatial registration of two omics datasets using feature-guided optimal transport and anchor-based alignment.

    This function aligns the spatial coordinates of ``adata_omics1`` to ``adata_omics2`` by first computing
    a fused Gromov-Wasserstein optimal transport plan that incorporates both modality-specific embeddings
    and shared features, then identifying anchor correspondences, and finally estimating a geometric
    transformation (e.g., affine) to warp the coordinates of omics1 into the space of omics2.

    Parameters
    ----------
    adata_omics1 : anndata.AnnData
        First omics AnnData object to be aligned (source). Must contain:
        - ``obsm['embedding']``: modality-specific feature embedding.
        - ``obsm['share_feature']``: cross-modality shared features.
        - ``obsm[use_obsm]``: original spatial coordinates.
    adata_omics2 : anndata.AnnData
        Second omics AnnData object (target/reference). Same requirements as above.
    alpha : float, optional (default: 1.0)
        Weight for modality-specific features in the fused Gromov-Wasserstein distance.
    beta : float, optional (default: 0.8)
        Weight for shared features in the fused distance.
    n_iter : int, optional (default: 10)
        Number of iterations for anchor finding/refinement.
    method : str, optional (default: 'affine')
        Type of geometric transformation to estimate (e.g., 'affine', 'rigid', 'similarity').
        Passed to the anchor-finding function.
    use_obsm : str, optional (default: 'spatial')
        Key in ``.obsm`` containing the original spatial coordinates to be transformed.
    new_obsm : str, optional (default: 'spatial_reg')
        Key under which the registered (transformed) coordinates will be stored in ``adata_omics1``.
    random_state : int, optional (default: 2030)
        Random seed for reproducibility in optimal transport.

    Returns
    -------
    tuple
        - adata_omics1 : anndata.AnnData
          Copy of the input ``adata_omics1`` with added ``obsm[new_obsm]`` containing registered coordinates.
        - registering_parameters : list
          List containing [T, omics1_index, omics2_index], where:
          - T : transformation matrix (or parameters).
          - omics1_index : indices of anchor spots in omics1.
          - omics2_index : indices of corresponding anchor spots in omics2.

    """
    feature_omics1 = adata_omics1.obsm['embedding']
    feature_omics2 = adata_omics2.obsm['embedding']
    share_feature_omics1 = adata_omics1.obsm['share_feature']
    share_feature_omics2 = adata_omics2.obsm['share_feature']
    spatial_omics1 = adata_omics1.obsm[use_obsm]
    spatial_omics2 = adata_omics2.obsm[use_obsm]

    sim_matrix = FGW_OT(
        feature_omics1 = feature_omics1,
        feature_omics2 = feature_omics2,
        share_feature_omics1 = share_feature_omics1,
        share_feature_omics2 = share_feature_omics2,
        spatial_omics1 = spatial_omics1,
        spatial_omics2 = spatial_omics2,
        a = alpha,
        b = beta,
        random_state=random_state,
    )

    T, omics1_index, omics2_index = find_anchor(
        spatial_omics1 = spatial_omics1,
        spatial_omics2 = spatial_omics2,
        sim_matrix = sim_matrix,
        n_iter=n_iter,
        method=method
    )

    adata_omics1.obsm[new_obsm] = apply_transform(T, adata_omics1.obsm[use_obsm])

    registering_parameters = [T, omics1_index, omics2_index]

    return adata_omics1, registering_parameters


def registration_by_downsampling(
    adata_omics1: anndata.AnnData,
    adata_omics2: anndata.AnnData,
    adata_omics1_downsampled: anndata.AnnData,
    adata_omics2_downsampled: anndata.AnnData,
) -> Tuple[anndata.AnnData, anndata.AnnData]:
    """
    Propagate coarse-grained registration from downsampled bins back to full-resolution spatial coordinates.

    Parameters
    ----------
    adata_omics1 : anndata.AnnData
        Full-resolution AnnData for omics1 (to be aligned).
    adata_omics2 : anndata.AnnData
        Full-resolution AnnData for omics2 (reference).
    adata_omics1_downsampled : anndata.AnnData
        Downsampled version of omics1 with registered coordinates in ``obsm['spatial_reg']``.
    adata_omics2_downsampled : anndata.AnnData
        Downsampled version of omics2 (usually unchanged spatial coordinates).

    Returns
    -------
    tuple of (anndata.AnnData, anndata.AnnData)
        - Updated ``adata_omics1`` with:
          - ``obsm['spatial_reg_bin']``: alignment applied at bin level.
          - ``obsm['spatial_reg_raw']``: final propagated alignment at original resolution.
        - Updated ``adata_omics2`` with ``obsm['spatial_reg_bin']`` (typically identity or minor adjustment).
    """

    crd_raw_omics1 = adata_omics1.obsm['spatial'].copy()
    crd_raw_omics2 = adata_omics2.obsm['spatial'].copy()

    crd_raw_bin_omics1 = adata_omics1_downsampled.obsm['spatial_raw'].copy()
    crd_raw_bin_omics2 = adata_omics2_downsampled.obsm['spatial_raw'].copy()

    crd_bin_omics1 = adata_omics1_downsampled.obsm['spatial_reg'].copy()
    crd_bin_omics2 = adata_omics2_downsampled.obsm['spatial'].copy()

    T1 = compute_affine_transform(crd_raw_bin_omics1, crd_bin_omics1)
    T2 = compute_affine_transform(crd_raw_bin_omics2, crd_bin_omics2)

    adata_omics1.obsm['spatial_reg_bin'] = apply_transform(T1, crd_raw_omics1)
    adata_omics2.obsm['spatial_reg_bin'] = apply_transform(T2, crd_raw_omics2)

    T3 = compute_affine_transform(crd_bin_omics2, crd_raw_bin_omics2)

    adata_omics1.obsm['spatial_reg_raw'] = apply_transform(T3, adata_omics1.obsm['spatial_reg_bin'])
    adata_omics2.obsm['spatial_reg_raw'] = crd_raw_omics2

    return adata_omics1, adata_omics2