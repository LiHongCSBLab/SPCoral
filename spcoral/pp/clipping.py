import numpy as np
import anndata

from typing import Optional, Dict

__all__ = [
    'clipping_patch',
]

from ..utils import extract_spatial_region

def clipping_patch(
        adata_omics1: anndata.AnnData,
        adata_omics2: anndata.AnnData,
        x_clip: Optional[float] = None,
        y_clip: Optional[float] = None,
        x_num: Optional[int] = None,
        y_num: Optional[int] = None,
        retain_edge: Optional[float] = 0.1,
        min_cells: Optional[int] = 0,
        use_obsm: str = 'spatial',
) -> Dict:
    """
    Divide the overlapping spatial region of two spatial omics into a grid of patches.

    This function computes the overlapping spatial region between two AnnData objects, then tiles it into a regular grid
    based on either specified patch sizes (`x_clip`/`y_clip`) or number of patches (`x_num`/`y_num`). For each grid cell,
    sub-regions are extracted from both datasets. Only patches where both omics have at least `min_cells` cells are retained.
    Optional overlapping edges between adjacent patches can be preserved to reduce boundary effects.

    Parameters
    ----------
    adata_omics1 : anndata.AnnData
        First spatial omics AnnData object. Must contain spatial coordinates (default in ``obsm['spatial']``)
        and features (in ``obsm['feat']``).
    adata_omics2 : anndata.AnnData
        Second spatial omics AnnData object with the same requirements.
    x_clip : float, optional
        Physical width of each patch along the x-axis (in coordinate units). Takes precedence if provided;
        otherwise derived from `x_num`.
    y_clip : float, optional
        Physical height of each patch along the y-axis.
    x_num : int, optional
        Desired number of patches along the x-axis. Used to compute `x_clip` if `x_clip` is not provided.
    y_num : int, optional
        Desired number of patches along the y-axis.
    retain_edge : float, optional (default: 0.1)
        Fraction of patch size to retain as overlap on each side between adjacent patches (range [0, 1)).
        A value of 0.1 means 10% overlap on each side.
    min_cells : int, optional (default: 0)
        Minimum number of cells required in a patch for both omics. Patches with fewer cells in either dataset are discarded.
    use_obsm : str, optional (default: 'spatial')
        Key in ``.obsm`` where spatial coordinates are stored.

    Returns
    -------
    dict
        Dictionary containing:
        - 'feature_omics1' : int
          Number of features in the first omics dataset.
        - 'feature_omics2' : int
          Number of features in the second omics dataset.
        - 'x_clip' : float
          Actual patch width used along x.
        - 'y_clip' : float
          Actual patch height used along y.
        - 'x_num' : int
          Actual number of patches along x.
        - 'y_num' : int
          Actual number of patches along y.
        - 'x_retain' : float
          Actual overlap length along x.
        - 'y_retain' : float
          Actual overlap length along y.
        - 'adata_omics1_clip_dict' : dict
          Mapping from patch key ``"{col}_{row}"`` to the corresponding sub-AnnData for omics1.
        - 'adata_omics2_clip_dict' : dict
          Same as above but for omics2.
    """

    adata_omics1_clip_dict = {}
    adata_omics2_clip_dict = {}

    min_x = np.max([np.min(adata_omics1.obsm[use_obsm].T[0]), np.min(adata_omics2.obsm[use_obsm].T[0])])
    min_y = np.max([np.min(adata_omics1.obsm[use_obsm].T[1]), np.min(adata_omics2.obsm[use_obsm].T[1])])

    max_x = np.min([np.max(adata_omics1.obsm[use_obsm].T[0]), np.max(adata_omics2.obsm[use_obsm].T[0])])
    max_y = np.min([np.max(adata_omics1.obsm[use_obsm].T[1]), np.max(adata_omics2.obsm[use_obsm].T[1])])

    if x_num is None and x_clip is None:
        print('')
        return ValueError

    if y_num is None and y_clip is None:
        print('')
        return ValueError

    if x_num is None:
        x_num = int((max_x - min_x) // x_clip)
    if x_clip is None:
        x_clip = (max_x - min_x) / x_num
    if y_num is None:
        y_num = int((max_y - min_y) // y_clip)
    if y_clip is None:
        y_clip = (max_y - min_y) / y_num

    if retain_edge != None:
        x_retain = x_clip * retain_edge
        y_retain = x_clip * retain_edge
    else:
        x_retain = 0
        y_retain = 0

    for col in range(x_num):
        for row in range(y_num):

            x = (col * x_clip + min_x, (col+1) * x_clip + min_x)
            y = (row * y_clip + min_y, (row+1) * y_clip + min_y)

            adata_omics1_clip = extract_spatial_region(
                adata=adata_omics1, minx=x[0], miny=y[0], maxx=x[1], maxy=y[1], retainx=x_retain, retainy=y_retain,
                used_obsm=use_obsm
            )

            adata_omics2_clip = extract_spatial_region(
                adata=adata_omics2, minx=x[0], miny=y[0], maxx=x[1], maxy=y[1], retainx=x_retain, retainy=y_retain,
                used_obsm=use_obsm
            )

            if adata_omics1_clip.shape[0] < min_cells or adata_omics2_clip.shape[0] < min_cells:
                continue

            adata_omics1_clip_dict[str(col) + '_' + str(row)] = adata_omics1_clip
            adata_omics2_clip_dict[str(col) + '_' + str(row)] = adata_omics2_clip

    clip_results = {
        'feature_omics1': adata_omics1.obsm['feat'].shape[1],
        'feature_omics2': adata_omics2.obsm['feat'].shape[1],
        'x_clip': x_clip,
        'y_clip': y_clip,
        'x_num': x_num,
        'y_num': y_num,
        'x_retain': x_retain,
        'y_retain': y_retain,
        'adata_omics1_clip_dict': adata_omics1_clip_dict,
        'adata_omics2_clip_dict': adata_omics2_clip_dict,
    }

    return clip_results